"""CIFAR-10 ResNet-18 training and embedding helpers."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ResNet18C(nn.Module):
    """ResNet-18 with a CIFAR stem and 512-wide embeddings."""

    def __init__(self):
        super().__init__()
        self.net = resnet18(num_classes=10)
        self.net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def embed(self, x):
        net = self.net
        x = net.relu(net.bn1(net.conv1(x)))
        x = net.layer4(net.layer3(net.layer2(net.layer1(net.maxpool(x)))))
        return torch.flatten(net.avgpool(x), 1)

    def forward(self, x):
        return self.net.fc(self.embed(x))


def get_embeddings(
    model, X, batch_size: int = 512, device=None
) -> dict[str, np.ndarray]:
    """Return batched penultimate activations and logits."""
    device = device or _default_device()
    original_device = next(model.parameters()).device
    model.to(device).eval()
    penultimate, logits = [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], device=device)
            embedding = model.embed(batch)
            penultimate.append(embedding.cpu().numpy())
            logits.append(model.net.fc(embedding).cpu().numpy())
    model.to(original_device)
    return {
        "penultimate": np.concatenate(penultimate).astype(np.float32, copy=False),
        "logits": np.concatenate(logits).astype(np.float32, copy=False),
    }


def _augment(x, generator):
    padded = F.pad(x, (4, 4, 4, 4), mode="replicate")
    offsets = torch.randint(9, (len(x), 2), device=x.device, generator=generator)
    crops = padded.unfold(2, 32, 1).unfold(3, 32, 1)
    batch = torch.arange(len(x), device=x.device)
    x = crops[batch, :, offsets[:, 0], offsets[:, 1]]
    flip = torch.rand(len(x), device=x.device, generator=generator) < .5
    return torch.where(flip[:, None, None, None], x.flip(-1), x)


def train_resnet(
    X,
    y,
    idx,
    seed: int,
    epochs: int = 20,
    lr: float = .1,
    batch_size: int = 128,
    init_model=None,
    device=None,
) -> ResNet18C:
    """Train ResNet-18 with SGD, CIFAR augmentation, cosine decay, and AMP."""
    if init_model is None:
        torch.manual_seed(seed)
        model = ResNet18C()
    else:
        model = deepcopy(init_model)
    device = torch.device(device or _default_device())
    model.to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = nn.CrossEntropyLoss()
    X_train = torch.as_tensor(np.asarray(X[idx], dtype=np.float32))
    y_train = torch.as_tensor(np.asarray(y[idx], dtype=np.int64))
    shuffle_generator = torch.Generator().manual_seed(seed)
    augment_generator = torch.Generator(device=device).manual_seed(seed)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(epochs):
        total_loss = 0.
        for batch_idx in torch.randperm(
            len(y_train), generator=shuffle_generator
        ).split(batch_size):
            xb = _augment(X_train[batch_idx].to(device), augment_generator)
            yb = y_train[batch_idx].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(batch_idx)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1}: train loss {total_loss / len(y_train):.4f}", flush=True)

    return model.cpu().eval()


def evaluate(model, X, y, idx=None, batch_size: int = 512) -> float:
    """Return classification accuracy on all or selected rows."""
    if idx is None:
        idx = np.arange(len(y))
    predictions = get_embeddings(model, X[idx], batch_size)["logits"].argmax(1)
    return float(np.mean(predictions == np.asarray(y)[idx]))


def demo():
    from topotrace.cifar import load_cifar10

    X_train, y_train, X_test, y_test = load_cifar10()
    model = train_resnet(
        X_train, y_train, np.arange(10000), seed=0, epochs=2, lr=.01
    )
    assert evaluate(model, X_test, y_test) > .45


if __name__ == "__main__":
    demo()
