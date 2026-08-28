import torch

from sentiment_manifold.directions.das import directional_replace


def test_directional_replace_preserves_orthogonal_component():
    base = torch.tensor([[1.0, 5.0]])
    source = torch.tensor([[3.0, -8.0]])
    direction = torch.tensor([1.0, 0.0])
    patched = directional_replace(base, source, direction)
    torch.testing.assert_close(patched, torch.tensor([[3.0, 5.0]]))
