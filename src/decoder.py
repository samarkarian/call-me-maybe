import re
import numpy as np
from typing import Any


def get_valid_token_ids(
    generated: str, valid_names: list[str], vocab: dict[int, str]
) -> list[int]:
    """Return token IDs that are valid continuations of a function name.

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
        Example: "fn_greet"
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


def force_token(
    text: str,
    input_ids: list[int],
    model: Any,
    vocab: dict[int, str],
) -> None:
    """Force the generation of an exact string token by token.

    Modifies input_ids in place by appending the tokens of the exact string.
    Example:
        input_ids = [101, 202, 303]
        force_token('", "parameters": {', input_ids, model, vocab)
        input_ids = [101, 202, 303, 456, 789, 123, ...]

    Args:
        text: The exact string to generate.
        input_ids: The current token sequence, modified in place.
        model: The LLM instance.
        vocab: The inverted vocabulary mapping token ID to token string.
    """
    remaining = text
    while remaining:
        valid_ids = [
            token_id for token_id, token_str in vocab.items()
            if remaining.startswith(token_str)
        ]
        logits = model.get_logits_from_input_ids(input_ids)
        masked = [float('-inf')] * len(logits)
        for i in valid_ids:
            masked[i] = logits[i]
        idx = int(np.argmax(masked))
        token_str = vocab[idx]
        input_ids.append(idx)
        remaining = remaining[len(token_str):]


def generate_number(
    input_ids: list[int],
    model: Any,
    vocab: dict[int, str],
) -> str:
    """Generate a JSON number value using constrained decoding.

    Args:
        input_ids: The current token sequence, modified in place.
        model: The LLM instance.
        vocab: The inverted vocabulary mapping token ID to token string.

    Returns:
        The generated number as a string.
    """
    number_pattern = re.compile(r'^-?(\d+\.?\d*|\.\d+)$')
    generated = ""

    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        valid_ids = []
        for token_id, token_str in vocab.items():
            candidate = generated + token_str
            valid = re.match(r'^-?[\d.]*$', candidate)
            if valid and candidate not in ("-", "."):
                valid_ids.append(token_id)
        if not valid_ids:
            break
        masked = [float('-inf')] * len(logits)
        for i in valid_ids:
            masked[i] = logits[i]
        idx = int(np.argmax(masked))
        token_str = vocab[idx]

        candidate = generated + token_str
        if not re.match(r'^-?[\d.]*$', candidate):
            break

        generated += token_str
        input_ids.append(idx)

        if number_pattern.match(generated):
            if '.' in generated:
                break
            next_logits = model.get_logits_from_input_ids(input_ids)
            stop_ids = [
                token_id for token_id, token_str in vocab.items()
                if token_str in (',', '}', ' ', '\n')
            ]
            best_stop = max(stop_ids, key=lambda i: next_logits[i])
            best_any = int(np.argmax(next_logits))
            stop_close = next_logits[best_stop] > next_logits[best_any] - 1.0
            if best_any in stop_ids or stop_close:
                break

    return generated


def generate_string(
    input_ids: list[int],
    model: Any,
    vocab: dict[int, str],
) -> str:
    """Generate a JSON string value without surrounding quotes.

    Args:
        input_ids: The current token sequence, modified in place.
        model: The LLM instance.
        vocab: The inverted vocabulary mapping token ID to token string.

    Returns:
        The generated string value (without quotes).
    """
    quote_ids = {token_id for token_id, token_str in vocab.items() if '"' in token_str}
    generated = ""
    char_count = 0

    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        escaped = generated.endswith('\\')

        if char_count > 300:
            exact_quote_ids = [
                token_id for token_id, token_str in vocab.items()
                if token_str == '"'
            ]
            valid_ids = exact_quote_ids if exact_quote_ids else list(quote_ids)
        elif escaped:
            valid_ids = list(vocab.keys())
        else:
            valid_ids = [
                token_id for token_id in vocab
                if token_id not in quote_ids
            ]

        masked = [float('-inf')] * len(logits)
        for i in valid_ids:
            masked[i] = logits[i]

        best_non_quote = int(np.argmax(masked))
        best_quote = max(quote_ids, key=lambda i: logits[i])
        if not escaped and char_count <= 300 and logits[best_quote] >= masked[best_non_quote]:
            input_ids.append(best_quote)
            break

        idx = int(np.argmax(masked))
        token_str = vocab[idx]
        if not escaped and '"' in token_str:
            input_ids.append(idx)
            break
        generated += token_str
        char_count += len(token_str)
        input_ids.append(idx)

    return generated


def generate_boolean(
    input_ids: list[int],
    model: Any,
    vocab: dict[int, str],
) -> str:
    """Generate a JSON boolean value using constrained decoding.

    Args:
        input_ids: The current token sequence, modified in place.
        model: The LLM instance.
        vocab: The inverted vocabulary mapping token ID to token string.

    Returns:
        Either 'true' or 'false'.
    """
    generated = ""
    targets = ["true", "false"]

    while generated not in targets:
        logits = model.get_logits_from_input_ids(input_ids)
        valid_ids = [
            token_id for token_id, token_str in vocab.items()
            if any(t.startswith(generated + token_str) for t in targets)
        ]
        masked = [float('-inf')] * len(logits)
        for i in valid_ids:
            masked[i] = logits[i]
        idx = int(np.argmax(masked))
        generated += vocab[idx]
        input_ids.append(idx)

    return generated


def generate_null(
    input_ids: list[int],
    model: Any,
    vocab: dict[int, str],
) -> None:
    """Force the generation of the JSON null value.

    Args:
        input_ids: The current token sequence, modified in place.
        model: The LLM instance.
        vocab: The inverted vocabulary mapping token ID to token string.
    """
    force_token("null", input_ids, model, vocab)


def generate_arguments(
    input_ids: list[int],
    model: Any,
    vocab: dict[int, str],
    func_def: dict[str, Any],
) -> dict[str, Any]:
    """Generate JSON arguments for a function call using constrained decoding.

    Args:
        input_ids: Token sequence after function name, modified in place.
        model: The LLM instance.
        vocab: The inverted vocabulary mapping token ID to token string.
        func_def: The function definition dict containing 'parameters'.

    Returns:
        A dict mapping parameter names to their generated values.
    """
    parameters = func_def['parameters']
    result: dict[str, Any] = {}

    force_token('", "parameters": {', input_ids, model, vocab)

    param_items = list(parameters.items())
    for i, (param_name, param_info) in enumerate(param_items):
        param_type = param_info['type']

        force_token(f'"{param_name}": ', input_ids, model, vocab)

        if param_type == 'string':
            force_token('"', input_ids, model, vocab)
            value: Any = generate_string(input_ids, model, vocab)
        elif param_type == 'number':
            value = generate_number(input_ids, model, vocab)
            value = float(value)
        elif param_type == 'integer':
            value = generate_number(input_ids, model, vocab)
            value = int(float(value))
        elif param_type == 'boolean':
            raw = generate_boolean(input_ids, model, vocab)
            value = raw == 'true'
        else:
            generate_null(input_ids, model, vocab)
            value = None

        result[param_name] = value

        if i < len(param_items) - 1:
            force_token(', ', input_ids, model, vocab)

    force_token('}', input_ids, model, vocab)

    return result
