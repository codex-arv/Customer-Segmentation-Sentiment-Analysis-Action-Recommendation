# prediction route which runs the live inference pipeline: accepts title + review, returns structured prediction result

from fastapi import APIRouter, Depends, HTTPException
from models import PredictRequest, PredictResponse, ErrorResponse
from loader import ArtifactRegistry, get_artifacts
from predictor import predict_review

router = APIRouter(prefix = "/predict", tags = ["Prediction"])

@router.post(
    "",
    response_model  = PredictResponse,
    responses       = {
        422: {"model": ErrorResponse, "description" : "Validation error"},
        500: {"model": ErrorResponse, "description" : "Inference error"},
        503: {"model": ErrorResponse, "description" : "Artifacts not loaded"},
    },
    summary         = "Predict cluster, topic, sentiment and get LLM interpretation",
    description     = """
                        Runs a new customer review through the complete ML + LLM pipeline:

                        1. **Preprocessing** — Cleans title + review
                        2. **Embedding** — SentenceTransformer (all-mpnet-base-v2)
                        3. **Category prediction** — Logistic classifier on 768-dimensional embeddings
                        4. **Cluster prediction** — UMAP reduction → LinearSVC
                        5. **Super cluster lookup** — KMeans mapping
                        6. **Topic prediction** — BERTopic per category+super_cluster
                        7. **Sentiment prediction** — TF-IDF + LinearSVC
                        8. **LLM interpretation** — Groq API (llama-3.1-8b-instant)

                        Returns full hierarchical segmentation with business intelligence.
                       """
)
async def predict(
    request   : PredictRequest,
    artifacts : ArtifactRegistry = Depends(get_artifacts)
):
    # safety guard: ensure artifacts are loaded
    if not artifacts.is_ready:
        raise HTTPException(
            status_code  = 503,
            detail       = {
                "error"  : "Service unavailable",
                "detail" : "Artifacts are still loading or failed to load. Check '/health' for status.",
                "path"   : "/predict"
            }
        )

    # run inference
    try:
        result = predict_review(
            title     = request.title,
            text      = request.text,
            artifacts = artifacts
        )
    except ValueError as e:
        raise HTTPException(
            status_code  = 422,
            detail       = {
                "error"  : "Prediction validation error",
                "detail" : str(e),
                "path"   : "/predict"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code  = 500,
            detail       = {
                "error"  : "Inference pipeline error",
                "detail" : str(e),
                "path"   : "/predict"
            }
        )

    # validate result structure before pydantic serialization
    required_sections = ["input", "predictions","llm_interpretation", "segment_context"]
    missing = [s for s in required_sections if s not in result]
    if missing:
        raise HTTPException(
            status_code  = 500,
            detail       = {
                "error"  : "Incomplete prediction result",
                "detail" : f"Missing sections: {missing}",
                "path"   : "/predict"
            }
        )

    return result