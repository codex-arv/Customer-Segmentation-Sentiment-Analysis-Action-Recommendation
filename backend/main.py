# fastapi app: entry point for the backend server
# handles lifespan, CORS, routing, registration & global exception handling

import asyncio
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    ALLOWED_ORIGINS,
    PIPELINE_VERSION,
    LLM_VERSION,
)

from loader import load_all_artifacts
from download_artifacts import download_artifacts

from routes import (
    predict,
    segments,
    recommendations,
    health,
    categories,
)


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("main")


# ============================================================================
# APPLICATION INITIALIZATION STATE
# ============================================================================

ARTIFACTS_READY = False
ARTIFACT_INITIALIZATION_ERROR = None


# ============================================================================
# BACKGROUND APPLICATION INITIALIZATION
# ============================================================================

async def initialize_application():

    global ARTIFACTS_READY
    global ARTIFACT_INITIALIZATION_ERROR

    try:

        logger.info("=" * 75)
        logger.info("INITIALIZING CUSTOMER SEGMENTATION API")
        logger.info("=" * 75)

        start = time.time()


        # --------------------------------------------------------------------
        # STEP 1: DOWNLOAD ARTIFACTS FROM CLOUDFLARE R2
        # --------------------------------------------------------------------

        logger.info(
            "Downloading artifacts from Cloudflare R2..."
        )

        await asyncio.to_thread(
            download_artifacts
        )

        logger.info(
            "R2 artifact download completed."
        )


        # --------------------------------------------------------------------
        # STEP 2: LOAD ARTIFACTS INTO MEMORY
        # --------------------------------------------------------------------

        logger.info(
            "Loading ML artifacts..."
        )

        await asyncio.to_thread(
            load_all_artifacts
        )


        # --------------------------------------------------------------------
        # INITIALIZATION COMPLETE
        # --------------------------------------------------------------------

        elapsed = time.time() - start

        ARTIFACTS_READY = True
        ARTIFACT_INITIALIZATION_ERROR = None

        logger.info(
            "All services are up and running!"
        )

        logger.info(
            f"Initialization complete in {elapsed:.1f} sec."
        )

        logger.info(
            "API is fully ready to process requests."
        )


    except Exception as e:

        ARTIFACTS_READY = False
        ARTIFACT_INITIALIZATION_ERROR = str(e)

        logger.error(
            f"APPLICATION INITIALIZATION FAILED: {e}",
            exc_info=True
        )


print("APP IS STARTING NOW!")


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("=" * 75)
    logger.info("STARTING => Customer Segmentation API")
    logger.info(
        f"Pipeline: {PIPELINE_VERSION}, LLM: {LLM_VERSION}"
    )
    logger.info("=" * 75)

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT wait for initialization here.
    #
    # The initialization process downloads approximately 4 GB of artifacts
    # and loads multiple ML models.
    #
    # Running it as a background task allows Uvicorn to bind to PORT
    # immediately, preventing Render's port detection timeout.
    # ------------------------------------------------------------------------

    asyncio.create_task(
        initialize_application()
    )

    # Yield immediately.
    #
    # Uvicorn can now keep the application alive and listen on the Render port.
    #
    yield


    # ------------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------------

    logger.info(
        "Shutting down — Bye"
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST TIMING LOGGER
# ============================================================================

@app.middleware("http")
async def log_requests(
    request: Request,
    call_next
):

    start = time.time()

    response = await call_next(request)

    elapsed = (
        time.time() - start
    ) * 1000

    logger.info(
        f"{request.method} "
        f"{request.url.path} "
        f"→ {response.status_code} "
        f"({elapsed:.1f}ms)"
    )

    return response


# ============================================================================
# REQUEST VALIDATION ERROR HANDLER
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):

    errors = []

    for error in exc.errors():

        errors.append({
            "field": " → ".join(
                str(l)
                for l in error["loc"]
            ),
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


# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    logger.error(
        f"Unhandled exception on "
        f"{request.url.path}: {exc}",
        exc_info=True
    )


    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path)
        }
    )


# ============================================================================
# ROUTER REGISTRATION
# ============================================================================

app.include_router(
    predict.router
)

app.include_router(
    segments.router
)

app.include_router(
    recommendations.router
)

app.include_router(
    health.router
)

app.include_router(
    categories.router
)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get(
    "/",
    tags=["Root"],
    summary="API root — version and endpoint index"
)
async def root():

    return {

        "name": API_TITLE,

        "version": API_VERSION,

        "pipeline_version": PIPELINE_VERSION,

        "llm_version": LLM_VERSION,

        "status": (
            "ready"
            if ARTIFACTS_READY
            else "initializing"
        ),

        "artifacts_ready": ARTIFACTS_READY,

        "initialization_error": (
            ARTIFACT_INITIALIZATION_ERROR
        ),

        "endpoints": {

            "POST /predict":
                "Run inference on a new review",

            "GET /segments":
                "Get all segments with health status",

            "GET /segments?category={category}":
                "Filter segments by category",

            "GET /recommendations/{category}/{super_cluster}":
                "Get cached recommendations",

            "GET /health":
                "System health and artifact status",

            "GET /docs":
                "Interactive API documentation",

            "GET /redoc":
                "ReDoc API documentation",

        }
    }