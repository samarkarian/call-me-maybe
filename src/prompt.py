from typing import Any


def ft_prompt(prompt: dict[str, str], definition: list[Any]) -> str:
    """Build a prompt string with available functions and the user request.

    Args:
        prompt: A dict containing the user prompt under the key 'prompt'.
        definition: A list of function definition dicts.

    Returns:
        A formatted string ready to be encoded and passed to the LLM.
    """
    lines = [f"{d['name']}: {d['description']}" for d in definition]
    functions_str = "\n".join(lines)
    user_request = prompt['prompt']

    return (
        f"You are an expert AI routing agent.\n"
        f"Your task is to map the user's natural language input to the "
        f"exact function that can answer it.\n\n"
        f"AVAILABLE FUNCTIONS:\n{functions_str}\n\n"
        f"RULES:\n"
        f"1. Select the most appropriate function from the list above.\n"
        f"2. Extract the required parameters from the USER INPUT.\n"
        f"3. Output strictly valid JSON.\n"
        f"4. If a string is given, extract it from the quotes.\n\n"
        f'USER INPUT:\n"{user_request}"\n\n'
        f'{{"prompt":"{user_request}","name":"'
    )


def main_prompt(definition: list[Any], prompt: Any) -> str:
    """Build a prompt string for a given test entry and function definitions.

    Args:
        definition: A list of function definition dicts.
        prompt: A dict containing the user prompt under the key 'prompt'.

    Returns:
        A formatted string ready to be encoded and passed to the LLM.
    """

    encode = ft_prompt(prompt, definition)

    return encode
