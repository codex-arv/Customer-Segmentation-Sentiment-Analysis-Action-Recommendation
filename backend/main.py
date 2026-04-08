# fastapi app: deployment-safe version (no heavy loading at startup)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import logging

from config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    ALLOWED_ORIGINS,
    PIPELINE_VERSION,
    LLM_VERSION,
)

# 🚨 DO NOT import loader or heavy modules here
# from loader import load_all_artifacts
# from routes import predict, segments, recommendations, health, categories

print("🚀 APP IS STARTING NOW!")

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("main")

# app instance
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# ✅ ROOT (keep this simple)
@app.get("/")
def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "llm_version": LLM_VERSION,
        "status": "running",
        "message": "Backend deployed successfully 🚀"
    }

# ✅ HEALTH CHECK (important for Render)
@app.get("/health")
def health():
    return {"status": "healthy"}

# ✅ CORS middleware (safe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ request timing logger
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)"
    )
    return response

# ✅ validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " → ".join(str(l) for l in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    return JSONResponse(
        status_code=422,
        content={
            "error": "Request validation failed",
            "detail": errors,
            "path": str(request.url.path)
        }
    )

# ✅ global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path)
        }
    )