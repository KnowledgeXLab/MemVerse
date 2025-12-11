# orchestrator.py
import os
import json
import uuid
import logging
import logging.config
from pathlib import Path
import asyncio
from datetime import datetime
from openai import OpenAI

# ===== Load Your Modules =====
from MemoryKB.User_Conversation import process_image as pi
from MemoryKB.User_Conversation import process_video as pv
from MemoryKB.User_Conversation import process_audio as pa
from MemoryKB import build_memory as bm
from MemoryKB.Long_Term_Memory.Graph_Construction import lightrag_openai_demo as Lgraph
from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag import LightRAG, QueryParam
from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag.kg.shared_storage import initialize_pipeline_status
from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag.utils import logger, set_verbose_debug
from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag.kg.shared_storage import initialize_share_data

# ======================================================
#       Global Paths (MMKG-style)
# ======================================================
BASE_DIR = os.path.join("MemoryKB", "Long_Term_Memory", "Graph_Construction", "MMKG")
CORE_DIR = os.path.join(BASE_DIR, "core")
EPISODIC_DIR = os.path.join(BASE_DIR, "episodic")
SEMANTIC_DIR = os.path.join(BASE_DIR, "semantic")

MEMORY_JSON_DIR = os.path.join("MemoryKB", "Long_Term_Memory", "memory_chunks")
CORE_JSON = os.path.join(MEMORY_JSON_DIR, "core_memory.json")
EPISODIC_JSON = os.path.join(MEMORY_JSON_DIR, "episodic_memory.json")
SEMANTIC_JSON = os.path.join(MEMORY_JSON_DIR, "semantic_memory.json")

# ======================================================
#       Global Config / Storage
# ======================================================
ROOT = Path(__file__).parent
USER_CONV_DIR = ROOT / "MemoryKB" / "User_Conversation"
CONV_JSON = USER_CONV_DIR / "conversation.json"
USER_CONV_DIR.mkdir(parents=True, exist_ok=True)

MAIN_BASE_URL = os.getenv("OPENAI_BASE_URL")
MAIN_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=MAIN_API_KEY, base_url=MAIN_BASE_URL)

RAG_INITIALIZED = False
mem_core = None
mem_epi = None
mem_sem = None

# ======================================================
#       Logging
# ======================================================
def configure_logging():
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir = os.getenv("LOG_DIR", os.getcwd())
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "lightrag_demo.log")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {"formatter": "default", "class": "logging.StreamHandler"},
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": int(os.getenv("LOG_MAX_BYTES", 10485760)),
                    "backupCount": int(os.getenv("LOG_BACKUP_COUNT", 5)),
                    "encoding": "utf-8",
                },
            },
            "loggers": {"lightrag": {"handlers": ["console", "file"], "level": "INFO", "propagate": False}},
        }
    )
    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

# ======================================================
#       Conversation Utilities
# ======================================================
def append_to_conversation(entry: dict):
    if CONV_JSON.exists():
        try:
            data = json.loads(CONV_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    data.append(entry)
    CONV_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def generate_unique_id() -> str:
    time_part = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    uuid_part = str(uuid.uuid4())
    unique_id = f"{time_part}_{uuid_part}"
    return unique_id

# ======================================================
#       RAG Initialization
# ======================================================
async def initialize_single_rag(working_dir):
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    return rag

async def insert_chunks_from_json(rag: LightRAG, json_path: str):
    if not os.path.exists(json_path):
        print(f"⚠️ JSON chunk file does not exist: {json_path}")
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

async def initialize_rag():
    global RAG_INITIALIZED, mem_core, mem_epi, mem_sem
    configure_logging()
    initialize_share_data(workers=1) # Important!
    for d in [CORE_DIR, EPISODIC_DIR, SEMANTIC_DIR]:
        os.makedirs(d, exist_ok=True)

    mem_core = await initialize_single_rag(CORE_DIR)
    mem_epi = await initialize_single_rag(EPISODIC_DIR)
    mem_sem = await initialize_single_rag(SEMANTIC_DIR)

    await initialize_pipeline_status()

# ======================================================
#       File Saving (Multi-modal)
# ======================================================
def save_file(file_bytes, filename, subdir):
    path = USER_CONV_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_bytes)
    return path

# ======================================================
#       Long-term Memory Update
# ======================================================
async def update_long_term_memory(entry):
    try:
        bm.process_memory([entry])
    except Exception as e:
        print(f"⚠️ Error processing memory: {e}")

    await insert_chunks_from_json(mem_core, bm.memory_files.get("MemoryKB/Long_Term_Memory/system/core_memory_agent.txt", ""))
    await insert_chunks_from_json(mem_epi, bm.memory_files.get("MemoryKB/Long_Term_Memory/system/episodic_memory_agent.txt", ""))
    await insert_chunks_from_json(mem_sem, bm.memory_files.get("MemoryKB/Long_Term_Memory/system/semantic_memory_agent.txt", ""))

# ======================================================
#       Parametric Memory / RAG Query
# ======================================================
async def call_parametric_memory(query: str):
    try:
        PM_BASE_URL = os.getenv("PM_BASE_URL")
        PM_API_KEY  = os.getenv("PM_API_KEY")
        client_pm = OpenAI(api_key=PM_API_KEY, base_url=PM_BASE_URL)

        pm_completion = await client_pm.chat.completions.create(
            model="parametric-memory",
            messages=[
                {"role": "system", "content": "You are a parametric memory generator."},
                {"role": "user", "content": query}
            ]
        )
        return pm_completion.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ Parametric memory failed: {e}"

async def check_pm_relevance(query: str, pm_memory: str) -> bool:
    try:
        relevance_check = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer ONLY 'yes' or 'no'. Determine if memory is relevant to query."
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nMemory: {pm_memory}\nRelevant?"
                }
            ]
        )
        decision = relevance_check.choices[0].message["content"].strip().lower()
        return "yes" in decision
    except Exception:
        return False

async def rag_retrieve(query: str, mode: str = "hybrid"):
    try:
        return await mem_core.aquery(query, param=QueryParam(mode=mode))
    except Exception as e:
        return f"⚠️ RAG retrieval failed: {e}"

async def generate_final_answer(query: str, memory: str):
    prompt = f"""
You are the user's memory assistant.

User query:
{query}

Relevant memory:
{memory}

If memory is insufficient, answer normally.
"""
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with long-term memory."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ LLM generation failed: {e}"

# ======================================================
#       Handlers for FastAPI
# ======================================================
async def handle_insert(query, video, audio, image):
    entry_id = generate_unique_id()
    entry = {"id": entry_id, "query": query, "videocaption": None, "audiocaption": None, "imagecaption": None}

    if video:
        video_path = save_file(await video.read(), video.filename, "video")
        entry["videocaption"] = pv.process_video(video_path, MAIN_BASE_URL, MAIN_API_KEY)
    if audio:
        audio_path = save_file(await audio.read(), audio.filename, "audio")
        entry["audiocaption"] = pa.process_audio(audio_path, MAIN_BASE_URL, MAIN_API_KEY)
    if image:
        image_path = save_file(await image.read(), image.filename, "image")
        entry["imagecaption"] = pi.process_image(image_path, MAIN_BASE_URL, MAIN_API_KEY)

    append_to_conversation(entry)
    await update_long_term_memory(entry)
    return entry

async def handle_query(query: str, mode: str, use_pm: bool):
    pm_memory, pm_relevant = None, False
    if use_pm:
        pm_memory = await call_parametric_memory(query)
        pm_relevant = await check_pm_relevance(query, pm_memory)

    rag_memory = None
    if not pm_relevant:
        rag_memory = await rag_retrieve(query, mode=mode)

    memory_text = ""
    if pm_relevant:
        memory_text += f"[Parametric Memory]\n{pm_memory}\n\n"
    if rag_memory:
        memory_text += f"[Long-term Memory]\n{rag_memory}\n"

    final_answer = await generate_final_answer(query, memory_text)
    return {
        "query": query,
        "mode": mode,
        "pm_used": use_pm,
        "pm_memory": pm_memory,
        "pm_relevant": pm_relevant,
        "rag_memory": rag_memory,
        "final_answer": final_answer,
    }
