import json
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from core.loop_controller import run_loop

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    description: str
    custom_cases: list = []
    existing_prompt: str = ""


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "frontend", "index.html"))


@app.post("/generate")
async def generate(request: GenerateRequest):
    async def event_stream():
        async for update in run_loop(request.description, request.custom_cases or None, request.existing_prompt or None):
            yield {"data": json.dumps(update)}

    return EventSourceResponse(event_stream())


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
