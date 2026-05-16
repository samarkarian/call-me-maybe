import sys
from parser import main_parser

def main() -> None:
    main_parser()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execution interrupted by the user.")
        sys.exit(1)