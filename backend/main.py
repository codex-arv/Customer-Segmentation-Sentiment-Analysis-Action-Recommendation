# fastapi app: entry point for the backend server, handles lifespan, CORS, routing, registration & global exception handlign

from contextlib import asynccontextmanager
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
from loader import load_all_artifacts
from routes import predict, segments, recommendations, health, categories

# logging setup
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("main")

# lifespan: startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("="*75)
    logger.info("STARTING => Customer Segmentation API")
    logger.info(f"Pipeline: {PIPELINE_VERSION}, LLM: {LLM_VERSION}")
    logger.info("="*75)

    start = time.time()

    try:
        load_all_artifacts()
        elapsed = time.time() - start
        logger.info("All services are up and running!")
        logger.info(f"Startup complete in {elapsed:.1f}sec.")
    except Exception as e:
        logger.error(f"STARTUP FAILED: {e}")
        raise

    yield 

    # shutdown
    logger.info("Shutting down — Bye")

# app instance
app = FastAPI(
    title       = API_TITLE,
    description = API_DESCRIPTION,
    version     = API_VERSION,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    openapi_url = "/openapi.json",
)

# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

# request timing logger (useful for monitoring inference latency)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = (time.time() - start) * 1000  # ms

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)"
    )
    return response

# global exception handling
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request : Request,
    exc     : RequestValidationError
):
    """
    Handles Pydantic validation errors — e.g. missing fields,
    wrong types, constraint violations in request body.
    Returns clean 422 with field-level error details.
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field"   : " → ".join(str(l) for l in error["loc"]),
            "message" : error["msg"],
            "type"    : error["type"]
        })

    return JSONResponse(
        status_code  = 422,
        content      = {
            "error"  : "Request validation failed",
            "detail" : errors,
            "path"   : str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(
    request : Request,
    exc     : Exception
):
    """
    Catches any unhandled exception — prevents raw Python
    tracebacks from leaking to the frontend.
    """
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code  = 500,
        content      = {
            "error"  : "Internal server error",
            "detail" : str(exc),
            "path"   : str(request.url.path)
        }
    )

#router registration
app.include_router(predict.router)
app.include_router(segments.router)
app.include_router(recommendations.router)
app.include_router(health.router)
app.include_router(categories.router)

# root endpoint
@app.get(
    "/",
    tags    = ["Root"],
    summary = "API root — version and endpoint index"
)
async def root():
    return {
        "name"             : API_TITLE,
        "version"          : API_VERSION,
        "pipeline_version" : PIPELINE_VERSION,
        "llm_version"      : LLM_VERSION,
        "status"           : "running",
        "endpoints"        : {
        "POST /predict"                                    : "Run inference on a new review",
        "GET  /segments"                                   : "Get all segments with health status",
        "GET  /segments?category={category}"               : "Filter segments by category",
        "GET  /recommendations/{category}/{super_cluster}" : "Get cached recommendations",
        "GET  /health"                                     : "System health and artifact status",
        "GET  /docs"                                       : "Interactive API documentation",
        "GET  /redoc"                                      : "ReDoc API documentation",
        }
    }