"""
FastAPI server for FinSage RAG Agent.
Replaces the previous Flask app; preserves all routes, auth, SSE streaming, and DB behavior.
"""
import atexit
import datetime
import json
import logging
import os
import shutil
import signal
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Timer
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env from project root so CONFIG_PATH, DATABASE_URL, PORT, BEARER_TOKEN, etc. are set
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import requests
import yaml
from fastapi import FastAPI, Request, Depends, Body, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates

from database import close_db, get_session, init_db
from load_document import LoadDataRequest, load_data_into_collection
from models import Feedback
from utils.ragManager import RAGManager
from utils.vllmChatService import ChatService
from treerag.tree_rag_service import TreeRagService
from services.feedback_processor import process_feedback_records

# ---------------------------------------------------------------------------
# Config and DB (used in lifespan and routes)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "server.log"
BACKUP_DIR = Path("/root/autodl-tmp/server_logs")


# ---------------------------------------------------------------------------
# OpenAPI request/response schemas (for spec and validation)
# ---------------------------------------------------------------------------

class ApiChatRequest(BaseModel):
    """Request body for synchronous and streaming chat."""
    collection_name: str = Field(..., description="Collection to query")
    question: str = Field(..., description="User question")
    session_id: Optional[str] = Field(None, description="Session ID; generated if omitted")
    internal_input: Optional[Any] = Field(None, description="Optional internal assistant context")
    interrupt_index: Optional[int] = Field(None, description="Optional interrupt index for editing")
    strategy: Optional[str] = Field(
        None,
        description="Optional strategy override. Use 'treerag' to enable TreeRAG for this request.",
    )
    treerag_max_depth: Optional[int] = Field(
        None,
        ge=0,
        description="Optional TreeRAG max depth override when strategy='treerag'.",
    )


class ApiChatResponseData(BaseModel):
    response: str
    session_id: str
    history: list = Field(default_factory=list)


class InternalAssistantRequest(BaseModel):
    """Request body for adding an internal assistant message to a session."""
    collection_name: str = Field(..., description="Collection to operate on")
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Internal assistant message")


class ReportErrorRequest(BaseModel):
    """Request body for reporting a client error."""
    collection_name: str = Field(..., description="Collection to operate on")
    session_id: str = Field(..., description="Session ID")
    error_message: str = Field(..., description="Error description from the client")


class SubmitRatingRequest(BaseModel):
    """Request body for submitting feedback rating for a response."""
    collection_name: str = Field(..., description="Collection to operate on")
    session_id: str = Field(..., description="Session ID")
    response_id: str = Field(..., description="Response ID from the chat stream")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    question: str = Field(..., description="Original question")
    response_content: str = Field(..., description="The response text being rated")
    feedback: Optional[str] = Field(None, description="Optional free-text feedback")
    user: Optional[str] = Field(None, description="Optional user identifier")


class ApiEnvelope(BaseModel):
    """Standard API response envelope."""
    status: str = Field(..., description="'success' or 'error'")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Any] = Field(None, description="Payload when status is success")


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        config = {}
    # Merge LLM/model defaults from config/llm.yaml so all model keys are defined in one place
    config_dir = Path(config_path).parent
    llm_path = config_dir / "llm.yaml"
    if llm_path.exists():
        with open(llm_path, encoding="utf-8") as f:
            llm = yaml.safe_load(f)
        if llm:
            for k, v in llm.items():
                config.setdefault(k, v)

    # Runtime override keys from environment (.env is loaded at module import).
    # Example: doc_processing_base_url <- env DOC_PROCESSING_BASE_URL.
    env_override_keys = [
        "doc_processing_base_url",
        "doc_processing_provider",
        "doc_processing_embeddings_provider",
        "feedback_classifier_provider",
        "treerag_llm_provider",
        "test_llm_api_key",
    ]
    for key in env_override_keys:
        val = config.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            env_name = key.upper()
            env_val = os.getenv(env_name)
            if env_val is not None and env_val != "":
                config[key] = env_val
    return config


def get_config_path() -> Path:
    raw = os.getenv("CONFIG_PATH")
    if raw:
        return Path(raw)
    return PROJECT_ROOT.parent / "config" / "production.yaml"


# ---------------------------------------------------------------------------
# Response envelope (same as Flask: status, message, data)
# ---------------------------------------------------------------------------

def success_response(data=None, message: str = "Success", status_code: int = 200):
    return JSONResponse(
        status_code=status_code,
        content={"status": "success", "message": message, "data": data},
    )


def error_response(message: str = "An error occurred", data=None, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "message": message, "data": data},
    )


def validate_collection_or_error(request: Request, collection_name: str) -> JSONResponse | None:
    name = (collection_name or "").strip()
    if not name:
        return error_response(message="collection_name not provided", status_code=400)
    rag_manager = request.app.state.rag_manager
    if not rag_manager.has_collection(name):
        return error_response(message=f"Collection '{name}' does not exist", status_code=404)
    return None


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

security = HTTPBearer(auto_error=False)


async def require_bearer_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """Validate Bearer token; return JSONResponse with status/message envelope on failure (same as Flask)."""
    token = getattr(request.app.state, "bearer_token", None)
    if not token:
        return error_response(message="Bearer token not configured", status_code=401)
    if not credentials:
        return error_response(message="Missing Authorization header", status_code=401)
    if not credentials.credentials or credentials.credentials != token:
        return error_response(message="Invalid bearer token", status_code=401)
    return credentials.credentials


async def check_token_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    """For /api/check_token: validate Bearer; return JSONResponse with envelope on failure."""
    token = getattr(request.app.state, "bearer_token", None)
    if not token:
        return error_response(message="Bearer token not configured", status_code=401)
    if not credentials:
        return error_response(message="Missing Authorization header", status_code=401)
    if not credentials.credentials or credentials.credentials != token:
        return error_response(message="Invalid bearer token", status_code=401)


# ---------------------------------------------------------------------------
# Helpers used by routes (low-rating webhook and SSE parsing)
# ---------------------------------------------------------------------------

def extract_reply_contents(sse_text: str) -> str:
    """Extract payload.content from event:reply in SSE text. Returns last reply."""
    contents = []
    current_event = None
    data_buffer = []
    for line in sse_text.splitlines():
        if line.startswith("event:"):
            if current_event == "reply" and data_buffer:
                data_json = json.loads("".join(data_buffer))
                contents.append(data_json["payload"]["content"])
            current_event = line[6:].strip()
            data_buffer = []
        elif line.startswith("data:"):
            data_buffer.append(line[5:].strip())
    if current_event == "reply" and data_buffer:
        data_json = json.loads("".join(data_buffer))
        contents.append(data_json["payload"]["content"])
    return contents[-1] if contents else ""


def handle_low_rating(
    config: dict,
    session_id: str,
    feedback: str,
    question: str,
    response: str,
) -> str | None:
    appkey = config.get("r1_online_appkey")
    url = config.get("r1_online_url")
    if not appkey or not url:
        return None
    content = (
        f'User Issues：{question}\nCurrent Answer： {response}\n'
        "User is not satisfied with the current answer, please search the internet and judge"
        if not feedback
        else (
            f'User Issues：{question}\nCurrent Answer： {response}\n'
            f"User is not satisfied with the current answer, this is the user's feedback on the answer: {feedback}\n"
            "Please search the internet and judge"
        )
    )
    payload = {
        "session_id": session_id,
        "bot_app_key": appkey,
        "visitor_biz_id": session_id,
        "content": content,
        "incremental": True,
        "streaming_throttle": 10,
        "visitor_labels": [],
        "custom_variables": {},
        "search_network": "enable",
    }
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=False,
        timeout=None,
    )
    resp.encoding = "utf-8"
    return extract_reply_contents(resp.text)


# ---------------------------------------------------------------------------
# Lifespan: config, logging, RAG/ChatService, cleanup timer, DB, signal handlers
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    config_path = get_config_path()
    config = load_config(str(config_path))

    bearer_token = config.get("bearer_token") or os.getenv("BEARER_TOKEN")
    if not bearer_token:
        raise ValueError("Bearer token not configured")
    app.state.bearer_token = bearer_token
    app.state.config = config

    log_level = config.get("log_level", "WARNING")
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        filename=str(LOG_PATH),
        filemode="w",
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    app.state.logger = logger

    def backup_log():
        for h in logger.handlers:
            h.flush()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(LOG_PATH, BACKUP_DIR / f"server{ts}.log")
            print(f"Log backed up to {BACKUP_DIR}/server{ts}.log")
        except Exception as e:
            print(f"Backup failed: {e}")

    atexit.register(backup_log)

    def graceful_exit(signum, frame):
        logger.warning("Received signal %s, backing up log before exit", signum)
        backup_log()
        logging.shutdown()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        signal.signal(sig, graceful_exit)

    rag_manager = RAGManager(config=config, collections=None)
    configured_collections = config.get("collections_top_k")
    if isinstance(configured_collections, dict) and configured_collections:
        for collection_name, top_k in configured_collections.items():
            if int(top_k) <= 0:
                continue
            rag_manager.create_collection(collection_name)
            rag_manager.upsert_collection_retriever(collection_name, int(top_k))
    else:
        default_top_k = int(config.get("default_collection_top_k", 10))
        rag_manager.hydrate_from_chroma(default_top_k=default_top_k)
    chat_service = ChatService(
        config=config, rag_manager=rag_manager, rerank_topk=config["rerank_topk"]
    )
    app.state.rag_manager = rag_manager
    app.state.chat_service = chat_service
    app.state.tree_rag_service = TreeRagService(config=config, rag_manager=rag_manager, chat_service=chat_service)

    def schedule_cleanup():
        chat_service.cleanup_old_sessions()
        try:
            app.state.tree_rag_service.cleanup_old_sessions()
        except Exception as e:
            logger.exception("TreeRAG cleanup error: %s", e)
        t = Timer(300, schedule_cleanup)
        t.daemon = True
        t.start()
        app.state.cleanup_timer = t
        logger.info("Scheduled next session cleanup in 5 minutes")

    cleanup_timer = Timer(300, schedule_cleanup)
    cleanup_timer.daemon = True
    cleanup_timer.start()
    app.state.cleanup_timer = cleanup_timer
    logger.info("Initial session cleanup scheduled in 5 minutes")

    database_url = os.getenv("DATABASE_URL") or config.get("database_url")
    if not database_url:
        raise ValueError("database_url not configured (set DATABASE_URL or database_url in config)")
    init_db(database_url)

    # Optional: periodic feedback processing (classify + alias records)
    feedback_stop = threading.Event()
    feedback_interval = max(60, int(config.get("feedback_processing_interval_seconds", 600)))
    feedback_enabled = config.get("feedback_processing_enabled", True)
    last_id_path = config.get("feedback_last_processed_id_file") or str(PROJECT_ROOT.parent / "log" / "feedback_last_processed_id.txt")
    categories_path = config.get("feedback_categories_path") or str(PROJECT_ROOT.parent / "config" / "feedback_categories.json")

    def _feedback_loop():
        while not feedback_stop.is_set():
            if feedback_stop.wait(timeout=feedback_interval):
                break
            if feedback_stop.is_set():
                break
            try:
                process_feedback_records(config, last_id_path, categories_path=categories_path)
            except Exception as e:
                logger.exception("Feedback processor error: %s", e)

    if feedback_enabled:
        feedback_thread = threading.Thread(target=_feedback_loop, daemon=True, name="feedback_processor")
        feedback_thread.start()
        app.state.feedback_stop = feedback_stop
        app.state.feedback_thread = feedback_thread
        logger.info("Feedback processor started (interval=%ds)", feedback_interval)
    else:
        app.state.feedback_stop = None
        app.state.feedback_thread = None

    yield

    # Shutdown: stop feedback processor, cancel cleanup timer, close DB
    if getattr(app.state, "feedback_stop", None) is not None:
        app.state.feedback_stop.set()
    # Shutdown: cancel initial and any rescheduled cleanup timer
    cleanup_timer.cancel()
    current = getattr(app.state, "cleanup_timer", None)
    if current is not None and current is not cleanup_timer:
        current.cancel()
    close_db()


# ---------------------------------------------------------------------------
# App and middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinSage RAG Agent",
    description="Multi-retrieval RAG API — chat, streaming, feedback, and internal assistant.",
    version="0.1.0",
    lifespan=lifespan,
    tags_metadata=[
        {"name": "Health", "description": "Liveness and readiness."},
        {"name": "Auth", "description": "Token validation."},
        {"name": "Chat", "description": "Synchronous and streaming Q&A."},
        {"name": "Ingestion", "description": "Collection and chunk ingestion for RAG."},
        {"name": "Feedback", "description": "Ratings and feedback UI."},
        {"name": "Internal", "description": "Internal assistant and error reporting."},
        {"name": "Logs", "description": "Session logs."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


# ---------------------------------------------------------------------------
# Exception handler (same envelope as Flask)
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = getattr(request.app.state, "logger", logging.getLogger(__name__))
    logger.error("An unexpected error occurred: %s", exc)
    return error_response(
        message=f"Internal Server Error: {str(exc)}",
        status_code=500,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"], summary="Health check")
async def health_check():
    return success_response(message="Server is running")


@app.get("/api/check_token", dependencies=[Depends(check_token_optional)], tags=["Auth"], summary="Validate bearer token")
async def check_token():
    return success_response(message="Token is valid")


@app.post(
    "/load-data",
    tags=["Ingestion"],
    summary="Create/load collection data into Chroma + BM25",
    response_model=ApiEnvelope,
)
async def load_data(
    body: LoadDataRequest,
    request: Request,
    _: str = Depends(require_bearer_token),
):
    try:
        rag_manager = request.app.state.rag_manager
        config = request.app.state.config
        result = load_data_into_collection(rag_manager, config, body)
        return success_response(data=result, message="Data loaded successfully")
    except Exception as e:
        request.app.state.logger.error("An error occurred in /load-data endpoint: %s", e)
        return error_response(message=str(e))


@app.post(
    "/api_chat",
    tags=["Chat"],
    summary="Synchronous Q&A",
    response_model=ApiEnvelope,
    responses={400: {"description": "Bad request (e.g. question not provided)"}},
)
async def api_chat(
    body: ApiChatRequest,
    request: Request,
    _: str = Depends(require_bearer_token),
):
    try:
        collection_error = validate_collection_or_error(request, body.collection_name)
        if collection_error:
            return collection_error
        session_id = body.session_id or str(uuid.uuid4())
        if not body.question:
            return error_response(message="Question not provided")

        chat_service = request.app.state.chat_service
        strategy = (body.strategy or "").strip().lower()
        treerag_enabled = bool(request.app.state.config.get("treerag_enabled", False))
        use_treerag = strategy == "treerag" or (not strategy and treerag_enabled)

        if use_treerag:
            tree_service = request.app.state.tree_rag_service
            response_text, _root = tree_service.run(
                question=body.question,
                session_id=session_id,
                collection_name=body.collection_name,
                max_depth=body.treerag_max_depth,
            )
            # Keep existing session/history behavior for UI parity
            chat_manager = chat_service.get_or_create_chat_manager(session_id, body.collection_name)
            chat_manager.add_to_qa_history(body.question, response_text)
            history = chat_manager.qa_history
        else:
            response_text, _, _, _, _, _, history = chat_service.generate_response_async(
                body.question, session_id, body.collection_name, body.internal_input, body.interrupt_index
            )
        return success_response(
            data={
                "response": response_text,
                "session_id": session_id,
                "history": history,
            }
        )
    except Exception as e:
        request.app.state.logger.error("An error occurred in /api_chat endpoint: %s", e)
        return error_response(message=str(e))


def _stream_chat(app_ref, question, session_id, collection_name, internal_input, interrupt_index):
    """Generator for SSE: yields chunks, then saves Q&A to DB. Uses app_ref.state for DB and logger."""
    full_response = ""
    response_id = str(uuid.uuid4())
    chat_service = app_ref.state.chat_service
    stream_gen = chat_service.generate_response_async_stream(
        question, session_id, collection_name, internal_input, interrupt_index
    )
    for chunk in stream_gen:
        if full_response == "":
            try:
                chunk_data = chunk.replace("data: ", "").strip()
                chunk_json = json.loads(chunk_data)
                if "response" in chunk_json:
                    chunk_json["question"] = question
                    chunk_json["response_id"] = response_id
                    full_response += chunk_json["response"]
                    chunk = f"data: {json.dumps(chunk_json)}\n\n"
            except Exception as e:
                app_ref.state.logger.error("Error modifying first chunk: %s", e)
        else:
            try:
                chunk_data = chunk.replace("data: ", "").strip()
                chunk_json = json.loads(chunk_data)
                if "response" in chunk_json:
                    full_response += chunk_json["response"]
            except Exception as e:
                app_ref.state.logger.error("Error parsing chunk: %s", e)
        yield chunk

    try:
        chat_manager = chat_service.get_or_create_chat_manager(session_id, collection_name)
        log = chat_manager.get_runtime_log()
        with get_session() as session:
            session.add(
                Feedback(
                    session_id=session_id,
                    response_id=response_id,
                    rating=0,
                    question=question,
                    response=full_response,
                    is_rag=1,
                    log=json.dumps(log),
                )
            )
        app_ref.state.logger.info("Saved Q&A pair to database with response_id: %s", response_id)
    except Exception as e:
        app_ref.state.logger.error("Error saving Q&A to database: %s", e)


@app.post(
    "/api_chat_stream",
    tags=["Chat"],
    summary="Streaming Q&A (SSE)",
    responses={
        200: {"description": "SSE stream of chunks", "content": {"text/event-stream": {}}},
        400: {"description": "Bad request"},
    },
)
async def api_chat_stream(
    body: ApiChatRequest,
    request: Request,
    _: str = Depends(require_bearer_token),
):
    try:
        collection_error = validate_collection_or_error(request, body.collection_name)
        if collection_error:
            return collection_error
        session_id = body.session_id or str(uuid.uuid4())
        if not body.question:
            return error_response(message="Question not provided")

        strategy = (body.strategy or "").strip().lower()
        treerag_enabled = bool(request.app.state.config.get("treerag_enabled", False))
        use_treerag = strategy == "treerag" or (not strategy and treerag_enabled)
        if use_treerag:
            return error_response(
                message="TreeRAG strategy is not supported for streaming yet. Use /api_chat with strategy='treerag'.",
                status_code=400,
            )

        return StreamingResponse(
            _stream_chat(
                request.app,
                body.question,
                session_id,
                body.collection_name,
                body.internal_input,
                body.interrupt_index,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        request.app.state.logger.error(
            "An error occurred in /api_chat_stream endpoint: %s", e
        )
        return error_response(message=str(e))


@app.get("/test_api_chat", tags=["Chat"], summary="Test chat UI (returns HTML)")
async def test_api_chat(request: Request, collection_name: str = Query(..., description="Collection name")):
    collection_error = validate_collection_or_error(request, collection_name)
    if collection_error:
        return collection_error
    session_id = str(uuid.uuid4())
    chat_service = request.app.state.chat_service
    _ = chat_service.get_or_create_chat_manager(session_id, collection_name)
    return templates.TemplateResponse(
        "test_api.html",
        {"request": request, "session_id": session_id},
    )


@app.get("/feedback", tags=["Feedback"], summary="Feedback list UI (returns HTML)")
async def feedback(request: Request):
    try:
        with get_session() as session:
            rows = (
                session.query(Feedback)
                .order_by(Feedback.id.desc())
                .all()
            )
        feedback_list = [
            {
                "session_id": r.session_id,
                "time": r.created_at,
                "rating": r.rating,
                "feedback": (r.feedback or ""),
                "question": r.question,
                "response": r.response,
                "log": r.log,
                "user": (r.user or ""),
            }
            for r in rows
        ]
        return templates.TemplateResponse(
            "feedback.html",
            {"request": request, "feedbacks": feedback_list},
        )
    except Exception as e:
        request.app.state.logger.error("An error occurred in /feedback endpoint: %s", e)
        return error_response(message=str(e), status_code=500)


@app.post(
    "/api/internal_assistant",
    tags=["Internal"],
    summary="Add internal assistant message to a session",
    response_model=ApiEnvelope,
)
async def internal_assistant(
    body: InternalAssistantRequest,
    request: Request,
    _: str = Depends(require_bearer_token),
):
    try:
        collection_error = validate_collection_or_error(request, body.collection_name)
        if collection_error:
            return collection_error
        if not body.session_id:
            return error_response(message="Session ID not provided")
        if not body.message:
            return error_response(message="Internal assistant message not provided")

        chat_service = request.app.state.chat_service
        chat_manager = chat_service.get_or_create_chat_manager(body.session_id, body.collection_name)
        chat_manager.add_internal_assitant_message(body.message)
        return success_response(
            data={
                "session_id": body.session_id,
                "status": "Internal assistant message added successfully",
            }
        )
    except Exception as e:
        request.app.state.logger.error(
            "An error occurred in /api/internal_assistant endpoint: %s", e
        )
        return error_response(message=str(e))


@app.get(
    "/api/log",
    tags=["Logs"],
    summary="Get runtime log for a session",
    response_model=ApiEnvelope,
    responses={400: {"description": "Session ID not provided"}},
)
async def get_log(
    request: Request,
    collection_name: str = Query(..., description="Collection name"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    _: str = Depends(require_bearer_token),
):
    try:
        collection_error = validate_collection_or_error(request, collection_name)
        if collection_error:
            return collection_error
        if not session_id:
            return error_response(message="Session ID not provided")
        chat_service = request.app.state.chat_service
        chat_manager = chat_service.get_or_create_chat_manager(session_id, collection_name)
        logs = chat_manager.get_runtime_log()
        request.app.state.logger.info("%s", logs)
        return success_response(data=logs)
    except Exception as e:
        request.app.state.logger.error("An error occurred in /api/log endpoint: %s", e)
        return error_response(message=str(e))


@app.get(
    "/api/treerag/tree",
    tags=["Chat"],
    summary="Get last TreeRAG tree for a session",
    response_model=ApiEnvelope,
)
async def treerag_tree(
    request: Request,
    collection_name: str = Query(..., description="Collection name"),
    session_id: str = Query(..., description="Session ID"),
    _: str = Depends(require_bearer_token),
):
    collection_error = validate_collection_or_error(request, collection_name)
    if collection_error:
        return collection_error
    tree = request.app.state.tree_rag_service.get_tree(collection_name=collection_name, session_id=session_id)
    if not tree:
        return error_response(message="No TreeRAG tree found for this session", status_code=404)
    return success_response(data=tree)


@app.get(
    "/api/treerag/node",
    tags=["Chat"],
    summary="Get details for a TreeRAG node",
    response_model=ApiEnvelope,
)
async def treerag_node(
    request: Request,
    collection_name: str = Query(..., description="Collection name"),
    session_id: str = Query(..., description="Session ID"),
    node_id: str = Query(..., description="Node ID"),
    _: str = Depends(require_bearer_token),
):
    collection_error = validate_collection_or_error(request, collection_name)
    if collection_error:
        return collection_error
    node = request.app.state.tree_rag_service.get_node(
        collection_name=collection_name, session_id=session_id, node_id=node_id
    )
    if not node:
        return error_response(message="TreeRAG node not found", status_code=404)
    return success_response(data=node)


@app.post(
    "/api/report_error",
    tags=["Internal"],
    summary="Report client error and save session log",
    response_model=ApiEnvelope,
)
async def report_error(
    body: ReportErrorRequest,
    request: Request,
    _: str = Depends(require_bearer_token),
):
    try:
        collection_error = validate_collection_or_error(request, body.collection_name)
        if collection_error:
            return collection_error
        if not body.session_id:
            return error_response(message="Session ID not provided")
        if not body.error_message:
            return error_response(message="Error message not provided")

        chat_service = request.app.state.chat_service
        chat_manager = chat_service.get_or_create_chat_manager(body.session_id, body.collection_name)
        logs = chat_manager.get_runtime_log()

        error_log_dir = PROJECT_ROOT.parent / "log" / "error"
        error_log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        error_report = {
            "timestamp": ts,
            "session_id": body.session_id,
            "error_message": body.error_message,
            "session_log": logs,
        }
        error_file_path = error_log_dir / f"{ts}.json"
        with open(error_file_path, "w", encoding="utf-8") as f:
            json.dump(error_report, f, indent=2, ensure_ascii=False)
        request.app.state.logger.warning("Error report saved to %s", error_file_path)
        return success_response(message="Error report submitted successfully")
    except Exception as e:
        request.app.state.logger.error(
            "An error occurred in /api/report_error endpoint: %s", e
        )
        return error_response(message=str(e))


@app.post(
    "/api/submit_rating",
    tags=["Feedback"],
    summary="Submit rating and optional feedback for a response",
    response_model=ApiEnvelope,
    responses={400: {"description": "Missing required fields or invalid rating"}},
)
async def submit_rating(
    body: SubmitRatingRequest,
    request: Request,
    _: str = Depends(require_bearer_token),
):
    try:
        collection_error = validate_collection_or_error(request, body.collection_name)
        if collection_error:
            return collection_error
        if not body.session_id or not body.response_id or not body.question or not body.response_content:
            return error_response(message="Missing required fields")

        chat_service = request.app.state.chat_service
        chat_manager = chat_service.get_or_create_chat_manager(body.session_id, body.collection_name)
        log = chat_manager.get_runtime_log()
        config = request.app.state.config

        with get_session() as session:
            row = (
                session.query(Feedback)
                .filter(
                    Feedback.session_id == body.session_id,
                    Feedback.response_id == body.response_id,
                )
                .first()
            )
            if row:
                row.rating = body.rating
                row.feedback = body.feedback
                row.question = body.question
                row.response = body.response_content
                row.log = json.dumps(log)
                row.user = body.user or ""
            else:
                session.add(
                    Feedback(
                        session_id=body.session_id,
                        response_id=body.response_id,
                        rating=body.rating,
                        feedback=body.feedback,
                        question=body.question,
                        response=body.response_content,
                        log=json.dumps(log),
                        user=body.user or "",
                    )
                )

        online_answer = None
        if body.rating <= 2:
            online_answer = handle_low_rating(
                config, body.session_id, body.feedback or "", body.question, body.response_content
            )

        return success_response(
            data={"online_answer": online_answer},
            message="Rating submitted successfully",
        )
    except Exception as e:
        request.app.state.logger.error(
            "An error occurred in /api/submit_rating endpoint: %s", e
        )
        return error_response(message=str(e))


# ---------------------------------------------------------------------------
# Entrypoint (uvicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "6005"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
