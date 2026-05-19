import numpy as np
from typing import Any


def get_valid_token_ids(
    generated: str, valid_names: list[str], vocab: dict[int, str]
) -> list[int]:
    """Return token IDs that are valid continuations of at least one function name.

    Args:
        generated: The function name prefix generated so far.
        valid_names: The list of valid function names.
        vocab: The inverted vocabulary mapping token ID to token string.

    Returns:
        A list of token IDs that keep the generation on a valid path.
    """
    valid_ids = []
    for token_id, token_str in vocab.items():
        candidate = generated + token_str
        for name in valid_names:
            if name.startswith(candidate):
                valid_ids.append(token_id)
                break
    return valid_ids


def ft_decoder(
    input_ids: list[int],
    model: Any,
    valid_names: list[str],
    vocab: dict[int, str],
) -> str:
    """Generate a function name using constrained decoding.

    At each step, only tokens that are valid continuations of a known function
    name are allowed. All other logits are set to -inf before taking argmax.

    Args:
        input_ids: The encoded prompt as a list of token IDs.
        model: The LLM instance.
        valid_names: The list of valid function names to constrain generation.
        vocab: The inverted vocabulary mapping token ID to token string.

    Returns:
        The selected function name.
    """
    raw_logits = model.get_logits_from_input_ids(input_ids)
    generated_name = ""

    while generated_name not in valid_names:
        valid_ids = get_valid_token_ids(generated_name, valid_names, vocab)

        if not valid_ids:
            break

        masked_logits = [float('-inf')] * len(raw_logits)
        for i in valid_ids:
            masked_logits[i] = raw_logits[i]

        idx = int(np.argmax(masked_logits))
        token_str = vocab[idx]
        generated_name += token_str
        input_ids.append(idx)
        raw_logits = model.get_logits_from_input_ids(input_ids)

    return generated_name
