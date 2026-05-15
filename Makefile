MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs
VENV = .venv
LIB_LLM_SDK = llm_sdk

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	@rm -rf $$(find . -type d -name "__pycache__") $$(find . -type d -name ".mypy_cache")

lint:
	uv run python -m flake8 . --extend-exclude $(VENV),$(LIB_LLM_SDK)
	uv run python -m mypy . $(MYPY_FLAGS)

.PHONY: install run debug clean lint
