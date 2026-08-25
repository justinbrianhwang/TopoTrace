import numpy as np
import torch


class MLP(torch.nn.Module):
    """Two-hidden-layer MLP mapping 2D points to binary logits."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)


def get_embeddings(model: MLP, X: np.ndarray) -> dict[str, np.ndarray]:
    """Return post-ReLU hidden activations and logits for ``X``."""
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32)
        h1 = torch.relu(model.fc1(x))
        h2 = torch.relu(model.fc2(h1))
        logits = model.fc3(h2)
    return {name: value.numpy() for name, value in locals().items() if name in ("h1", "h2", "logits")}
