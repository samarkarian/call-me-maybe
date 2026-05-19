import sys
from src.parser import main_parser
from src.prompt import main_prompt
from src.model import get_model, get_vocab
from src.decoder import ft_decoder


def main() -> None:
    definition, calling_tests = main_parser()
    model = get_model()
    vocab = get_vocab(model)
    valid_names = [d['name'] for d in definition]
    prompt_str = main_prompt(definition, calling_tests)
    input_ids = model.encode(prompt_str)[0].tolist()
    result = ft_decoder(input_ids, model, valid_names, vocab)
    print(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execution interrupted by the user.")
        sys.exit(1)