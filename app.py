# app.py
from fastapi import FastAPI, UploadFile, Form
from orchestrator import (
    initialize_rag,
    handle_insert,
    handle_query
)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await initialize_rag()

@app.post("/insert")
async def insert(query: str = Form(...), video: UploadFile = None, audio: UploadFile = None, image: UploadFile = None):
    entry = await handle_insert(query, video, audio, image)
    return {"status": "ok", "entry": entry}

@app.post("/query")
async def query_api(query: str = Form(...), mode: str = Form("hybrid"), use_pm: bool = Form(False)):
    result = await handle_query(query, mode, use_pm)
    return {"status": "ok", **result}
