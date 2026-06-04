import logging
import sys

from src.answerer import generate_answer
from src.config import Config
from src.memory_loader import LoadResult, load_memories
from src.query_parser import parse_query
from src.retriever import retrieve
from src.utils import setup_logging

logger = logging.getLogger("memory_search.main")

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})


def run_repl(config: Config, load_result: LoadResult) -> None:
    print(
        f"\nLoaded {load_result.valid_count} memory records from "
        f"{config.memory_jsonl_path}\n"
    )
    print("Ask a memory question, or type 'q' to quit:")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in QUIT_COMMANDS:
            print("Goodbye.")
            break

        try:
            _handle_question(config, load_result, question)
        except Exception:
            logger.exception("Unexpected error while handling question")
            print("Something went wrong processing that question. Please try again.\n")


def _handle_question(
    config: Config,
    load_result: LoadResult,
    question: str,
) -> None:
    query = parse_query(question, config)
    results = retrieve(load_result.memories, query, config)
    answer = generate_answer(query, results, config)
    print(f"\n{answer}\n")


def main() -> None:
    setup_logging()

    try:
        config = Config.from_env()
        config.validate()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    load_result = load_memories(config)
    run_repl(config, load_result)


if __name__ == "__main__":
    main()
