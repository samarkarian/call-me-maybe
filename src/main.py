import sys
import json
import argparse
from pathlib import Path
from src.parser import main_parser
from src.prompt import main_prompt
from src.model import get_model, get_vocab
from src.decoder import ft_decoder, generate_arguments


def main() -> None:
    """Run the function-calling pipeline."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--functions_definition',
            default='data/input/functions_definition.json',
        )
        parser.add_argument(
            '--input',
            default='data/input/function_calling_tests.json',
        )
        parser.add_argument(
            '--output',
            default='data/output/function_calling_results.json',
        )
        args = parser.parse_args()

        definition, calling_tests = main_parser(
            args.functions_definition, args.input
        )
        model = get_model()
        vocab = get_vocab(model)
        valid_names = [d['name'] for d in definition]
        results = []
        for test in calling_tests:
            prompt_str = main_prompt(definition, test)
            input_ids = model.encode(prompt_str)[0].tolist()
            func_name = ft_decoder(input_ids, model, valid_names, vocab)
            func_def_list = [d for d in definition if d['name'] == func_name]
            if not func_def_list:
                print(f"Error: decoded function '{func_name}' not found in definitions.")
                sys.exit(1)
            func_def = func_def_list[0]
            arguments = generate_arguments(input_ids, model, vocab, func_def)
            results.append({
                'prompt': test['prompt'],
                'name': func_name,
                'parameters': arguments,
            })

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    except KeyboardInterrupt:
        print("Execution interrupted by the user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
