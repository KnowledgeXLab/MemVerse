import os
import sys
import asyncio
import logging
import logging.config
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug

BASE_DIR = os.path.join("MemoryKB", "Long_Term_Memory", "Graph_Construction", "MMKG")
CORE_DIR = os.path.join(BASE_DIR, "core")
EPISODIC_DIR = os.path.join(BASE_DIR, "episodic")
SEMANTIC_DIR = os.path.join(BASE_DIR, "semantic")

MEMORY_JSON_DIR = os.path.join("MemoryKB", "Long_Term_Memory", "memory_chunks")
CORE_JSON = os.path.join(MEMORY_JSON_DIR, "core_memory.json")
EPISODIC_JSON = os.path.join(MEMORY_JSON_DIR, "episodic_memory.json")
SEMANTIC_JSON = os.path.join(MEMORY_JSON_DIR, "semantic_memory.json")

def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir = os.getenv("LOG_DIR", os.getcwd())
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "lightrag_demo.log")

    print(f"\nLightRAG demo log file: {log_file_path}\n")

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"},
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


for d in [CORE_DIR, EPISODIC_DIR, SEMANTIC_DIR]:
    os.makedirs(d, exist_ok=True)


async def initialize_rag(working_dir):
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    return rag


async def insert_chunks_from_json(rag: LightRAG, json_path: str):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                text = chunk.get("output_text")
                if text:
                    await rag.ainsert(text)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON format at line {line_num}, skipping this line: {e}")
                continue

async def main():
    # Check if OPENAI_API_KEY environment variable exists
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    try:
        rag_core = await initialize_rag(CORE_DIR)
        rag_epi = await initialize_rag(EPISODIC_DIR)
        rag_sem = await initialize_rag(SEMANTIC_DIR)

        await initialize_pipeline_status()

        await insert_chunks_from_json(rag_core, CORE_JSON)
        await insert_chunks_from_json(rag_epi, EPISODIC_JSON)
        await insert_chunks_from_json(rag_sem, SEMANTIC_JSON)

        test_text = ["This is a test string for embedding."]
        embedding = await rag_core.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        query_text = "What are the top themes in this memory?"
        for rag, name in [(rag_core, "Core"), (rag_epi, "Episodic"), (rag_sem, "Semantic")]:
            print(f"\n=====================")
            print(f"Query mode: naive ({name})")
            print("=====================")
            print(await rag.aquery(query_text, param=QueryParam(mode="naive")))

            print(f"\n=====================")
            print(f"Query mode: local ({name})")
            print("=====================")
            print(await rag.aquery(query_text, param=QueryParam(mode="local")))

            print(f"\n=====================")
            print(f"Query mode: global ({name})")
            print("=====================")
            print(await rag.aquery(query_text, param=QueryParam(mode="global")))

            print(f"\n=====================")
            print(f"Query mode: hybrid ({name})")
            print("=====================")
            print(await rag.aquery(query_text, param=QueryParam(mode="hybrid")))

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # finalize storages
        for rag in [rag_core, rag_epi, rag_sem]:
            if rag:
                await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
