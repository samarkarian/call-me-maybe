from typing import Any


def ft_prompt(prompt: dict[str, str], definition: list[Any]) -> str:
    """Build a prompt string with available functions and the user request.

    Args:
        prompt: A dict containing the user prompt under the key 'prompt'.
        definition: A list of function definition dicts.

    Returns:
        A formatted string ready to be encoded and passed to the LLM.
    """
    lines = []
    for d in definition:
        params = ", ".join(
            f"{k}: {v['type']}" for k, v in d['parameters'].items()
        )
        lines.append(
            f"- {d['name']}({params})"
            f" -> {d['returns']['type']}: {d['description']}"
        )

    functions_str = "\n".join(lines)
    user_request = prompt['prompt']

    return (
        f"Available functions:\n{functions_str}\n\n"
        f"User request: {user_request}\n\n"
        f'Response (JSON):\n{{"name": "'
    )


def main_prompt(definition: list[Any], prompt: Any) -> str:
    """Run ft_prompt on a random prompt for testing purposes.

    Args:
        definition: A list of function definition dicts.
        prompt: A list of prompt dicts.
    """

    encode = ft_prompt(prompt, definition)

    return encode
