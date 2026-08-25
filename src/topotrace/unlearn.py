"""Machine-unlearning baselines for the MNIST pilot."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from topotrace.cnn import SmallCNN, evaluate, train_cnn


def _device() -> torch.device:
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        supported = [int(arch[3:].split("a")[0]) // 10
                     for arch in torch.cuda.get_arch_list()
                     if arch.startswith("sm_")]
        if not supported or major <= max(supported):
            return torch.device("cuda")
    return torch.device("cpu")


def _batches(idx, batch_size, *, shuffle=False, generator=None):
    idx = np.asarray(idx)
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=generator).numpy()]
    for start in range(0, len(idx), batch_size):
        yield idx[start : start + batch_size]


def _xy(X, y, idx, device):
    return (torch.as_tensor(X[idx], device=device),
            torch.as_tensor(y[idx], device=device, dtype=torch.long))


def _finish(model):
    return model.cpu().eval()


def _accuracy(model, X, y, idx, batch_size, device):
    correct = 0
    with torch.no_grad():
        for batch in _batches(idx, batch_size):
            xb, yb = _xy(X, y, batch, device)
            correct += int((model(xb).argmax(1) == yb).sum())
    return correct / len(idx)


def finetune(model, X, y, forget_idx, retain_idx, seed: int,
             epochs: int = 2, lr: float = 1e-4) -> SmallCNN:
    """Fine-tune a copy of ``model`` on retained examples."""
    return train_cnn(X, y, retain_idx, seed, epochs=epochs, lr=lr,
                     init_model=model, device=str(_device()))


def neggrad(model, X, y, forget_idx, retain_idx, seed: int,
            steps: int = 100, lr: float = 1e-4,
            batch_size: int = 128) -> SmallCNN:
    """Apply gradient ascent to cross-entropy on forgotten examples."""
    device = _device()
    student = deepcopy(model).to(device).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    step = 0
    while step < steps:
        for batch in _batches(forget_idx, batch_size, shuffle=True,
                              generator=generator):
            xb, yb = _xy(X, y, batch, device)
            optimizer.zero_grad()
            (-F.cross_entropy(student(xb), yb)).backward()
            optimizer.step()
            step += 1
            if step % max(1, (len(forget_idx) + batch_size - 1) // batch_size) == 0:
                student.eval()
                done = _accuracy(student, X, y, forget_idx, batch_size,
                                 device) < .05
                student.train()
                if done:
                    return _finish(student)
            if step == steps:
                break
    return _finish(student)


def _kl(student_logits, teacher_logits, temperature=2.0):
    student_logp = F.log_softmax(student_logits / temperature, dim=1)
    student_p = student_logp.exp()
    teacher_logp = F.log_softmax(teacher_logits / temperature, dim=1)
    return (student_p * (student_logp - teacher_logp)).sum(1).mean()


def scrub(model, X, y, forget_idx, retain_idx, seed: int,
          epochs: int = 2, lr: float = 1e-4, batch_size: int = 128,
          alpha: float = 0.5, gamma: float = 1.0) -> SmallCNN:
    """Run simplified max-forget/min-retain SCRUB updates."""
    device = _device()
    student = deepcopy(model).to(device)
    teacher = deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        student.train()
        for batch in _batches(forget_idx, batch_size, shuffle=True,
                              generator=generator):
            xb, _ = _xy(X, y, batch, device)
            optimizer.zero_grad()
            (-_kl(student(xb), teacher(xb))).backward()
            optimizer.step()
        for batch in _batches(retain_idx, batch_size, shuffle=True,
                              generator=generator):
            xb, yb = _xy(X, y, batch, device)
            optimizer.zero_grad()
            logits = student(xb)
            loss = gamma * F.cross_entropy(logits, yb)
            loss += alpha * _kl(logits, teacher(xb))
            loss.backward()
            optimizer.step()
    return _finish(student)


def _importance(model, X, y, idx, batch_size, max_batches, device):
    importance = [torch.zeros_like(p) for p in model.parameters()]
    count = 0
    model.train()
    for count, batch in enumerate(_batches(idx, batch_size), 1):
        if count > max_batches:
            count -= 1
            break
        xb, yb = _xy(X, y, batch, device)
        model.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        for total, parameter in zip(importance, model.parameters()):
            if parameter.grad is not None:
                total.add_(parameter.grad.square())
    return [value / count for value in importance]


def ssd(model, X, y, forget_idx, retain_idx,
        dampening_alpha: float = 10.0, lam: float = 1.0,
        batch_size: int = 128, max_batches: int = 50) -> SmallCNN:
    """Dampen parameters whose forget importance dominates retain importance."""
    device = _device()
    student = deepcopy(model).to(device)
    imp_f = _importance(student, X, y, forget_idx, batch_size, max_batches,
                        device)
    imp_r = _importance(student, X, y, retain_idx, batch_size, max_batches,
                        device)
    with torch.no_grad():
        for parameter, forget, retain in zip(student.parameters(), imp_f, imp_r):
            mask = forget > dampening_alpha * retain
            parameter[mask] *= torch.clamp(lam * retain[mask] / forget[mask],
                                            max=1.0)
    return _finish(student)


if __name__ == "__main__":
    from topotrace.mnist import load_mnist

    X_train, y_train, X_test, y_test = load_mnist()
    train_idx = np.arange(2000)
    forget_idx, retain_idx = train_idx[:100], train_idx[100:]
    original = train_cnn(X_train, y_train, train_idx, seed=0, epochs=1,
                         device=str(_device()))
    original_forget = evaluate(original, X_train, y_train, forget_idx)
    methods = {
        "finetune": finetune(original, X_train, y_train, forget_idx,
                             retain_idx, seed=0),
        "neggrad": neggrad(original, X_train, y_train, forget_idx,
                           retain_idx, seed=0),
        "scrub": scrub(original, X_train, y_train, forget_idx, retain_idx,
                       seed=0),
        "ssd": ssd(original, X_train, y_train, forget_idx, retain_idx),
    }
    for name, result in methods.items():
        assert any(not torch.equal(a, b) for a, b in
                   zip(original.parameters(), result.parameters())), name
    for name in ("finetune", "scrub"):
        assert evaluate(methods[name], X_train, y_train, retain_idx[:500]) > .5
    assert evaluate(methods["neggrad"], X_train, y_train,
                    forget_idx) < original_forget
    print("unlearn demo passed")
