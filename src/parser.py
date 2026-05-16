import json
import sys
from typing import Any
from pydantic import BaseModel, ValidationError, field_validator

VALID_TYPES = ['string', 'number', 'boolean', 'null']

class FieldType(BaseModel):
    """Represents a typed field in a function definition.

    Attributes:
        type: The JSON type of the field. Must be one of 'string',
            'number', 'boolean', or 'null'.
    """

    type: str

    @field_validator("type", mode="after")
    def check_type(cls, value: str) -> str:
        """Validate that the type is a supported JSON type.

        Args:
            value: The type string to validate.

        Returns:
            The validated type string.

        Raises:
            ValueError: If the type is not in the list of valid types.
        """
        if value not in VALID_TYPES:
            raise ValueError(
                f"Invalid value '{value}': must be one of {VALID_TYPES}"
            )
        return value


class DefinitionTest(BaseModel):
    """Represents a function definition entry.

    Attributes:
        name: The function name.
        description: A human-readable description of the function.
        parameters: A mapping of parameter names to their types.
        returns: The return type of the function.
    """

    name: str
    description: str
    parameters: dict[str, FieldType]
    returns: FieldType


class CallingTest(BaseModel):
    """Represents a single function-calling test prompt.

    Attributes:
        prompt: The natural language prompt to process.
    """

    prompt: str


def parse_calling_tests(calling_tests: list[Any]) -> None:
    """Validate each entry in the calling tests list against CallingTest.

    Args:
        calling_tests: A list of raw objects loaded from JSON.
    """
    for call in calling_tests:
        try:
            CallingTest.model_validate(call)
        except ValidationError as err:
            print(err)
            sys.exit(1)


def parse_definition(definition: list[Any]) -> None:
    """Validate each entry in the function definitions list against DefinitionTest.

    Args:
        definition: A list of raw objects loaded from JSON.
    """
    for defin in definition:
        try:
            DefinitionTest.model_validate(defin)
        except ValidationError as err:
            print(err)
            sys.exit(1)


def main_parser() -> None:
    """Load and validate the input JSON files for calling tests and definitions."""
    try:
        with open('./data/input/function_calling_tests.json') as file:
            calling_tests = json.load(file)
        with open('./data/input/functions_definition.json') as file:
            definition = json.load(file)
    except Exception as err:
        print(err)
        sys.exit(1)

    parse_calling_tests(calling_tests)
    parse_definition(definition)
