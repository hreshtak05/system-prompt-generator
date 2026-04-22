import json
import os
import base64
import tempfile
import pathlib
import asyncio
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from core.loop_controller import run_loop, run_test

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.ts', '.json', '.csv', '.html', '.css', '.xml', '.yaml', '.yml'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
IMAGE_MIME_TYPES = {'image/png': 'image/png', 'image/jpeg': 'image/jpeg', 'image/gif': 'image/gif', 'image/webp': 'image/webp'}


class ContextFile(BaseModel):
    filename: str
    content: str
    type: str = "text"


class GenerateRequest(BaseModel):
    description: str
    custom_cases: list = []
    existing_prompt: str = ""
    context_files: list = []  # List of {filename, content, type}


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "frontend", "index.html"))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Process uploaded file and extract its content."""
    content = await file.read()
    filename = file.filename or "unknown"
    mime_type = file.content_type or ""
    ext = pathlib.Path(filename).suffix.lower()

    if len(content) > MAX_FILE_SIZE:
        return {"error": f"File too large. Maximum size is 20MB."}

    try:
        # --- Text files: read directly ---
        if ext in TEXT_EXTENSIONS:
            try:
                text = content.decode("utf-8")
                return {"filename": filename, "content": text, "type": "text"}
            except UnicodeDecodeError:
                return {"error": "Could not read file as text. Make sure it is a plain text file."}

        # --- Images: describe with Gemini vision ---
        elif ext in IMAGE_EXTENSIONS or mime_type in IMAGE_MIME_TYPES:
            detected_mime = IMAGE_MIME_TYPES.get(mime_type, f"image/{ext[1:] if ext else 'jpeg'}")
            image_b64 = base64.b64encode(content).decode()
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content([
                {
                    "inline_data": {
                        "mime_type": detected_mime,
                        "data": image_b64
                    }
                },
                "Describe this image in full detail. Extract ALL visible text, UI elements, data, tables, charts, labels, and any other information. Be comprehensive — this description will be used to generate test cases for an AI assistant that processes this type of content."
            ])
            return {"filename": filename, "content": response.text, "type": "image"}

        # --- PDFs, Word docs, and other supported formats: use Gemini Files API ---
        else:
            tmp_path = None
            gemini_file = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".bin") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                gemini_file = genai.upload_file(tmp_path, display_name=filename)

                # Wait for Gemini to process the file
                for _ in range(30):
                    gemini_file = genai.get_file(gemini_file.name)
                    if gemini_file.state.name != "PROCESSING":
                        break
                    await asyncio.sleep(2)

                if gemini_file.state.name == "FAILED":
                    return {"error": "Gemini could not process this file type."}

                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content([
                    gemini_file,
                    "Extract and describe ALL content from this file comprehensively. Include all text, data, tables, structure, headings, and information. Be thorough — this will be used to generate specific test cases for an AI assistant."
                ])
                return {"filename": filename, "content": response.text, "type": "document"}

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if gemini_file:
                    try:
                        genai.delete_file(gemini_file.name)
                    except Exception:
                        pass

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}


@app.post("/generate")
async def generate(request: GenerateRequest):
    async def event_stream():
        async for update in run_loop(
            request.description,
            request.custom_cases or None,
            request.existing_prompt or None,
            request.context_files or None
        ):
            yield {"data": json.dumps(update)}

    return EventSourceResponse(event_stream(), ping=15)


@app.post("/test")
async def test_prompt(request: GenerateRequest):
    async def event_stream():
        if not request.existing_prompt or not request.existing_prompt.strip():
            yield {"data": json.dumps({"type": "error", "message": "No prompt provided to test."})}
            return
        async for update in run_test(
            request.description,
            request.existing_prompt.strip(),
            request.custom_cases or None,
            request.context_files or None
        ):
            yield {"data": json.dumps(update)}

    return EventSourceResponse(event_stream(), ping=15)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
