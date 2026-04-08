# returns full system status — artifact load state, model versions, segment and topic counts.

from fastapi import APIRouter, Depends
from models import HealthResponse, ArtifactStatus
from loader import ArtifactRegistry, get_artifacts
from config import (
    PIPELINE_VERSION,
    LLM_VERSION,
    CATEGORIES,
    BERTOPIC_MODEL_PATHS
)

router = APIRouter(
    prefix = "/health",
    tags   = ["Health"]
)

@router.get(
    "",
    response_model = HealthResponse,
    summary        = "Get system health and artifact status",
    description    = """
                        Returns the current status of the inference system:
                        - Whether all artifacts are loaded and ready
                        - Pipeline and LLM version numbers
                        - Active LLM model
                        - Count of loaded artifacts vs expected
                        - Per-artifact status details
                      """
)
async def get_health(
    artifacts: ArtifactRegistry = Depends(get_artifacts)
):
    # status list for each artifact
    artifact_details = []

    # heck each critical artifact
    checks = {
        "SentenceTransformer"   : artifacts.sentence_model,
        "UMAP model"            : artifacts.umap_model,
        "Inference scaler"      : artifacts.scaler_inference,
        "Inference PCA"         : artifacts.pca_inference,
        "Cluster predictor"     : artifacts.svc,
        "Category classifier"   : artifacts.cat_clf,
        "Sentiment classifier"  : artifacts.sentiment_clf,
        "TF-IDF vectorizer"     : artifacts.tfidf,
        "Cluster-to-super map"  : artifacts.cluster_to_super,
        "Topic cache"           : artifacts.topic_cache,
        "Super cluster cache"   : artifacts.super_cache,
        "Recommendations cache" : artifacts.reco_cache,
        "Enriched dataset"      : artifacts.df_final_clean,
        "LLM client"            : artifacts.llm_client,
    }

    for name, artifact in checks.items():
        status = "loaded" if artifact is not None else "missing"
        artifact_details.append(ArtifactStatus(
            name   = name,
            status = status,
            path   = ""
        ))

    # check bertopic models
    for sc_key in sorted(BERTOPIC_MODEL_PATHS.keys()):
        loaded = sc_key in artifacts.bertopic_models and artifacts.bertopic_models[sc_key] is not None
        artifact_details.append(ArtifactStatus(
            name   = f"BERTopic [{sc_key}]",
            status = "loaded" if loaded else "missing",
            path   = str(BERTOPIC_MODEL_PATHS[sc_key])
        ))

    loaded_count = sum(1 for a in artifact_details if a.status == "loaded")
    total_count  = len(artifact_details)

    return HealthResponse(
        status           = "ready" if artifacts.is_ready else "degraded",
        pipeline_version = PIPELINE_VERSION,
        llm_version      = LLM_VERSION,
        llm_model        = artifacts.llm_model or "not loaded",
        artifacts_loaded = loaded_count,
        artifacts_total  = total_count,
        categories       = CATEGORIES,
        total_segments   = len(artifacts.bertopic_models),
        total_topics     = len(artifacts.topic_cache),
        artifact_details = artifact_details
    )