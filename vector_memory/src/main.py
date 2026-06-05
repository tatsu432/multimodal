import logging
import sys

from src.answerer import generate_answer
from src.config import Config
from src.embedding_client import EmbeddingClient, create_embedding_client
from src.hybrid_retriever import hybrid_retrieve
from src.memory_loader import LoadResult, build_memories_by_id, load_memories
from src.query_parser import parse_query
from src.schema import LoadedMemory
from src.utils import setup_logging
from src.vector_store import VectorStore

logger = logging.getLogger("vector_memory.main")

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})


def run_repl(
    config: Config,
    load_result: LoadResult,
    memories_by_id: dict[str, LoadedMemory],
    vector_store: VectorStore,
    embedding_client: EmbeddingClient,
    indexed_count: int,
) -> None:
    print(f"\nLoaded {load_result.valid_count} memory records.")
    print(
        f"Indexed {indexed_count} memory records into Chroma collection "
        f"{config.chroma_collection_name}.\n"
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
            _handle_question(
                config,
                memories_by_id,
                question,
                vector_store,
                embedding_client,
            )
        except Exception:
            logger.exception("Unexpected error while handling question")
            print("Something went wrong processing that question. Please try again.\n")


def _handle_question(
    config: Config,
    memories_by_id: dict[str, LoadedMemory],
    question: str,
    vector_store: VectorStore,
    embedding_client: EmbeddingClient,
) -> None:
    query = parse_query(question, config)
    results = hybrid_retrieve(
        memories_by_id=memories_by_id,
        query=query,
        vector_store=vector_store,
        embedding_client=embedding_client,
        config=config,
    )
    answer = generate_answer(query, results, config)
    print(f"\n{answer}\n")


def _collection_count(vector_store: VectorStore) -> int:
    try:
        return vector_store._collection.count()
    except Exception:
        return 0


def main() -> None:
    setup_logging()

    try:
        config = Config.from_env()
        config.validate()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    load_result = load_memories(config)
    memories_by_id = build_memories_by_id(load_result.memories)

    try:
        embedding_client = create_embedding_client(config)
        vector_store = VectorStore(config, embedding_client)
        index_result = vector_store.index_memories(
            load_result.memories,
            rebuild=config.rebuild_index,
        )
    except RuntimeError as exc:
        print(f"Indexing error: {exc}", file=sys.stderr)
        sys.exit(1)

    indexed_count = _collection_count(vector_store)
    logger.info(
        "Index update: %d new, %d skipped (already indexed), %d total in collection",
        index_result.indexed,
        index_result.skipped_duplicate,
        indexed_count,
    )

    run_repl(
        config,
        load_result,
        memories_by_id,
        vector_store,
        embedding_client,
        indexed_count,
    )


if __name__ == "__main__":
    main()
