from cirilla.LLM_pieces import get_activation
import pytest
from requests import HTTPError

@pytest.mark.parametrize("activation_path", [
    "kernels-community/vllm-flash-attn3",
    "Motif-Technologies/activation",
    "motif-technologies/optimizer"
])
def test_get_activation_succeeds(activation_path):
    result = get_activation(activation_path)
    assert result is not None

@pytest.mark.parametrize("activation_path", [
    "frog/frog-frogger-dogger",
])
def test_get_activation_fails(activation_path):
    with pytest.raises(HTTPError):
        get_activation(activation_path)
