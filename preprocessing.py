import torch
import torch.nn.functional as F
from typing import Iterable


def encode_decision_history(history: Iterable[int], num_actions: int = 3) -> torch.Tensor:
    """Return a one-hot encoded tensor for a sequence of past actions.

    Parameters
    ----------
    history : iterable of int
        Sequence of past actions.
    num_actions : int, optional
        Number of discrete actions. Defaults to ``3``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, len(history) * num_actions)``.
    """
    hist = torch.tensor(list(history), dtype=torch.long)
    one_hot = F.one_hot(hist, num_classes=num_actions).float()
    return one_hot.view(1, -1)
