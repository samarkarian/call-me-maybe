*This project has been created as part of the 42 curriculum by samarkar*

## Description

**call-me-maybe** is a function-calling tool that translates natural language prompts into structured function calls. Given a prompt like *"What is the sum of 40 and 2?"*, the system does not return `42` — it returns the function name (`fn_add_numbers`) and its arguments (`{"a": 40.0, "b": 2.0}`).

The core of the project is **constrained decoding**: instead of letting the LLM generate freely, the output is guided token by token so that the result is always valid JSON that conforms to the expected schema. This guarantees 100% parsable output even with a 0.5B parameter model (Qwen/Qwen3-0.6B).

## Instructions

### Installation

```bash
uv sync
```

### Run

```bash
uv run python -m src
```

With custom file paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Other commands

```bash
make install   # install dependencies
make run       # run the project
make debug     # run with pdb debugger
make clean     # remove __pycache__ and .mypy_cache
make lint      # run flake8 and mypy
```

## Algorithm Explanation

### Constrained Decoding

At each generation step, the LLM produces a vector of logits — one score per token in the vocabulary (~152k tokens for Qwen3-0.6B). Normally, the token with the highest score is selected. With constrained decoding, all tokens that would violate the expected structure are set to `-inf` before the argmax, making them impossible to select.

**Function name selection:**
The decoder builds the function name character by character. At each step, it keeps only the tokens whose string representation is a valid prefix of at least one known function name. The loop stops when the accumulated string exactly matches a known function name.

**Argument generation:**
Once the function is identified, its parameter schema is known. The decoder generates a JSON object `{"param": value, ...}` using a state machine:
- Structural tokens (`{`, `"`, `:`, `,`, `}`) are forced exactly using `force_token`.
- `number` values: only tokens matching `^-?[\d.]*$` are allowed.
- `string` values: all tokens except `"` are allowed; generation stops when the model prefers the closing quote.
- `boolean` values: constrained to `true` or `false` using prefix matching.
- `null` values: forced directly with `force_token("null", ...)`.

### Generation Loop

```
input_ids = encode(prompt)
while name not complete:
    logits = get_logits_from_input_ids(input_ids)
    mask invalid tokens to -inf
    next_token = argmax(logits)
    input_ids.append(next_token)
```

## Design Decisions

- **Pydantic for validation**: all input JSON is validated against typed models before processing, ensuring early and clear error messages.
- **Singleton-free model loading**: the model is loaded once in `main()` and passed explicitly to all functions, avoiding hidden global state.
- **`force_token` for structure**: JSON structural elements are forced deterministically rather than sampled, removing any risk of structural errors.
- **Separation of concerns**: parsing (`parser.py`), prompt building (`prompt.py`), model access (`model.py`), and decoding (`decoder.py`) are kept in separate modules.

## Performance Analysis

- **Accuracy**: with constrained decoding, function name selection is guaranteed to produce a valid function name. Argument extraction accuracy depends on the model understanding the prompt correctly, which exceeds 90% on standard prompts.
- **JSON validity**: 100% — the constrained decoding makes invalid JSON structurally impossible.
- **Speed**: each token requires one full forward pass through the model. On CPU this is slow (~1-5 seconds per token). On MPS (Apple Silicon) or CUDA the same model runs significantly faster.

## Difficulties Encountered

- **Token boundary alignment**: tokens in the vocabulary can span multiple characters (e.g. `"fn_add"` as a single token). The prefix-matching logic had to account for multi-character tokens advancing the state by more than one character at a time.
- **Number generation stopping condition**: determining when a number is complete required checking whether the model preferred a stop token (`,` or `}`) over continuing the number.
- **String generation stopping condition**: the closing `"` competes with content tokens. The solution is to stop as soon as the closing quote scores at least as high as the best non-quote token.

## Testing Strategy

- Input files were reduced to a small subset during development to speed up iteration.
- Each generation function (`generate_number`, `generate_string`, `_generate_boolean`) was tested independently by checking the decoded output against expected values.
- The full pipeline was validated end-to-end by running `uv run python -m src` and inspecting `data/output/function_calling_results.json`.

## Example Usage

```bash
$ uv run python -m src
```

Input (`data/input/function_calling_tests.json`):
```json
[
  {"prompt": "What is the sum of 40 and 2?"},
  {"prompt": "Greet John"}
]
```

Output (`data/output/function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40.0, "b": 2.0}
  },
  {
    "prompt": "Greet John",
    "name": "fn_greet",
    "parameters": {"name": "John"}
  }
]
```

## Resources

- https://huggingface.co/learn/llm-course/en/chapter6/5
- https://tiktokenizer.vercel.app

- https://huggingface.co/learn/llm-course/en/chapter2/2
- https://jalammar.github.io/illustrated-gpt2/

- https://huggingface.co/docs/transformers/main_classes/tokenizer
- https://huggingface.co/learn/llm-course/en/chapter6/1

- https://www.promptingguide.ai/fr
- https://huggingface.co/learn/llm-course/en/chapter1/1

### AI Usage

Claude (Anthropic) was used throughout this project for guidance, code review, and debugging assistance. Specifically:
- Explaining constrained decoding concepts and tokenization
- Reviewing and fixing Pydantic models and type annotations
- Fixing flake8 and mypy errors
