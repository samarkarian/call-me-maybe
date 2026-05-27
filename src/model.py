import json
from llm_sdk import Small_LLM_Model  # type: ignore


def get_model() -> Small_LLM_Model:
    """Load and return a new LLM instance.

    Returns:
        A new Small_LLM_Model instance.
    """
    return Small_LLM_Model()


def get_vocab(model: Small_LLM_Model) -> dict[int, str]:
    """Load and return the inverted vocabulary (token ID -> token string).

    Args:
        model: The LLM instance to retrieve the vocab file path from.

    Returns:
        A dict mapping each token ID to its string representation.
    """
    bpe_replacements = {"Ġ": " ", "Ċ": "\n", "ĉ": "\t"}
    path = model.get_path_to_vocab_file()
    with open(path) as f:
        vocab = json.load(f)
    result = {}
    for token_str, token_id in vocab.items():
        clean = token_str
        for bpe_char, real_char in bpe_replacements.items():
            clean = clean.replace(bpe_char, real_char)
        result[token_id] = clean
    return result
