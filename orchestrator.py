# orchestrator.py
import os
import json
import uuid
from pathlib import Path
from openai import OpenAI

# ===== Load Your Modules =====
from MemoryKB.User_Conversation import process_image as pi
from MemoryKB.User_Conversation import process_video as pv
from MemoryKB.User_Conversation import process_audio as pa
from MemoryKB import build_memory as bm
from MemoryKB.Long_Term_Memory.Graph_Construction import lightrag_openai_demo as Lgraph
from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag import QueryParam


# ======================================================
#       GLOBAL CONFIG
# ======================================================
ROOT = Path(__file__).parent
USER_CONV_DIR = ROOT / "MemoryKB" / "User_Conversation"
CONV_JSON = USER_CONV_DIR / "conversation.json"
USER_CONV_DIR.mkdir(parents=True, exist_ok=True)

MAIN_BASE_URL = os.getenv("OPENAI_BASE_URL")
MAIN_API_KEY  = os.getenv("OPENAI_API_KEY")

# Main LLM client
client = OpenAI(api_key=MAIN_API_KEY, base_url=MAIN_BASE_URL)


# ======================================================
#       Conversation JSON Utilities
# ======================================================
def append_to_conversation(entry: dict):
    """Append a new entry to conversation.json."""
    if CONV_JSON.exists():
        try:
            data = json.loads(CONV_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(entry)
    CONV_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ======================================================
#      RAG Memory Initialization
# ======================================================
RAG_INITIALIZED = False
mem_core = None
mem_epi = None
mem_sem = None

async def initialize_rag():
    """Initialize the LightRAG memory structures."""
    global RAG_INITIALIZED, mem_core, mem_epi, mem_sem
    if not RAG_INITIALIZED:
        mem_core = await Lgraph.initialize_rag(Lgraph.CORE_DIR)
        mem_epi  = await Lgraph.initialize_rag(Lgraph.EPISODIC_DIR)
        mem_sem  = await Lgraph.initialize_rag(Lgraph.SEMANTIC_DIR)
        RAG_INITIALIZED = True

async def insert_chunks_from_file(file_path, mem_inst):
    """Insert file-based chunks into a RAG memory instance."""
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "output_text" not in item:
                    print(f"⚠️ Line {line_num} is missing 'output_text'")
                    continue
                chunk = {"output_text": item["output_text"]}
                await mem_inst.ainsert(chunk)
            except Exception as e:
                print(f"⚠️ Error inserting chunk at line {line_num}: {e}")


# ======================================================
#      Multi-Modal Processors (Encapsulated)
# ======================================================
def save_file(file_bytes, filename, subdir):
    path = USER_CONV_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_bytes)
    return path

# ======================================================
#      Memory Construction Wrapper
# ======================================================
async def update_long_term_memory(entry):
    try:
        bm.process_memory([entry])
    except Exception as e:
        print(f"⚠️ Error processing memory: {e}")
    # Insert chunks into RAG
    await insert_chunks_from_file(bm.memory_files["MemoryKB/Long_Term_Memory/system/core_memory_agent.txt"], mem_core)
    await insert_chunks_from_file(bm.memory_files["MemoryKB/Long_Term_Memory/system/episodic_memory_agent.txt"], mem_epi)
    await insert_chunks_from_file(bm.memory_files["MemoryKB/Long_Term_Memory/system/semantic_memory_agent.txt"], mem_sem)


# ======================================================
#      LLM & PM Query Encapsulation
# ======================================================
async def call_parametric_memory(query: str):
    """Call the parametric memory model."""
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

        pm_answer = pm_completion.choices[0].message["content"]
        return pm_answer

    except Exception as e:
        return f"⚠️ Parametric memory failed: {e}"


async def check_pm_relevance(query: str, pm_memory: str) -> bool:
    """Use GPT to determine whether PM is relevant."""
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
    """Unified RAG retrieval interface."""
    try:
        return await mem_core.aquery(query, param=QueryParam(mode=mode))
    except Exception as e:
        return f"⚠️ RAG retrieval failed: {e}"


# ======================================================
#      Final Answer Synthesis
# ======================================================
async def generate_final_answer(query: str, memory: str):
    """Ask the main LLM to produce final answer."""
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
#      Main High-Level Handlers Called in app.py
# ======================================================
async def handle_insert(query, video, audio, image):
    """Main entry for /insert route."""
    entry_id = str(uuid.uuid4())

    entry = {
        "id": entry_id,
        "query": query,
        "videocaption": None,
        "audiocaption": None,
        "imagecaption": None,
    }

    api_url = MAIN_BASE_URL
    api_key = MAIN_API_KEY

    # Multi-modal Processing
    if video:
        video_path = save_file(await video.read(), video.filename, "video")
        entry["videocaption"] = pv.process_video(video_path, api_url, api_key)

    if audio:
        audio_path = save_file(await audio.read(), audio.filename, "audio")
        entry["audiocaption"] = pa.process_audio(audio_path, api_url, api_key)

    if image:
        image_path = save_file(await image.read(), image.filename, "image")
        entry["imagecaption"] = pi.process_image(image_path, api_url, api_key)

    # Save conversation
    append_to_conversation(entry)

    # Memory update
    await update_long_term_memory(entry)

    return entry


async def handle_query(query: str, mode: str, use_pm: bool):
    """Main entry for /query route."""

    # ---- Parametric Memory ----
    pm_memory = None
    pm_relevant = False

    if use_pm:
        pm_memory = await call_parametric_memory(query)
        pm_relevant = await check_pm_relevance(query, pm_memory)

    # ---- RAG Retrieval ----
    rag_memory = None
    if not pm_relevant:
        rag_memory = await rag_retrieve(query, mode=mode)

    # ---- Combine memory ----
    memory_text = ""
    if pm_relevant:
        memory_text += f"[Parametric Memory]\n{pm_memory}\n\n"
    if rag_memory:
        memory_text += f"[Long-term Memory]\n{rag_memory}\n"

    # ---- Final Answer ----
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