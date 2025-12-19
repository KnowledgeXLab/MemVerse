# mcp_server.py
import asyncio
from typing import Optional
from fastmcp import FastMCP
from orchestrator import initialize_rag, handle_insert, handle_query

mcp = FastMCP("memverse")

_initialized = False
_lock = asyncio.Lock()

async def ensure_init():
    global _initialized
    if _initialized:
        return
    async with _lock:
        if not _initialized:
            await initialize_rag()
            _initialized = True
            print("MCP ready")

@mcp.tool()
async def insert_memory(
    query: str,
    image: Optional[str] = None,
    audio: Optional[str] = None,
    video: Optional[str] = None,
):
    await ensure_init()
    entry = await handle_insert(query, image, audio, video)
    return {"entry": entry}

@mcp.tool()
async def query_memory(
    query: str,
    mode: str = "hybrid",
    use_pm: bool = True,
):
    await ensure_init()
    return await handle_query(query, mode, use_pm)

if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=5250, transport="http")
