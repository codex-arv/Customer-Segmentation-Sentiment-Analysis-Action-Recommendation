# returns cached LLM generated recommendations for a specific category + super cluster segment

from fastapi import APIRouter, Depends, HTTPException, Path
from models import RecommendationResponse, ErrorResponse
from loader import ArtifactRegistry, get_artifacts
from config import CATEGORIES, normalize_category
import pandas as pd

router = APIRouter(
    prefix = "/recommendations",
    tags   = ["Recommendations"]
)

@router.get(
    "/{display_category}/{super_cluster_name}",
    response_model = RecommendationResponse,
    responses      = {
        404: {"model" : ErrorResponse, "description": "Segment not found"},
        422: {"model" : ErrorResponse, "description": "Invalid category or segment name"},
        503: {"model" : ErrorResponse, "description": "Artifacts not loaded"},
    },
    summary        = "Get cached recommendations for a specific segment by name",
    description    = """
                        Returns LLM-generated business recommendations for a specific
                        category + super_cluster segment using human-readable names.
                        Call GET /categories first to get valid category display names
                        and super cluster names for this endpoint.
                        Includes:
                        - Overall segment health
                        - Ranked recommendations with priority, impact, urgency
                        - Business opportunities (3 items)
                        - Risk flags (3 items)
                        - Sentiment distribution for context
                     """
)
async def get_recommendations(
    display_category   : str = Path(
        ...,
        description = "Display category name e.g. 'Electronics'",
        examples    = ["Electronics", "Grocery & Gourmet Food"]
    ),
    super_cluster_name : str = Path(
        ...,
        description = "Super cluster name e.g. 'Electronics Setup'",
        examples    = ["Electronics Setup", "Electronics Issues"]
    ),
    artifacts: ArtifactRegistry = Depends(get_artifacts)
):
    from config import DISPLAY_TO_CATEGORY

    if not artifacts.is_ready:
        raise HTTPException(
            status_code  = 503,
            detail       = {
                "error"  : "Service unavailable",
                "detail" : "Artifacts not loaded. Check /health.",
                "path"   : f"/recommendations/{display_category}/{super_cluster_name}"
            }
        )

    # convert display name to internal name
    internal_category = normalize_category(display_category)

    if internal_category not in CATEGORIES:
        raise HTTPException(
            status_code  = 422,
            detail       = {
                "error"  : "Invalid category",
                "detail" : (f"'{display_category}' is not a valid category. Call GET /categories for valid display names."),
                "path"   : f"/recommendations/{display_category}/{super_cluster_name}"
            }
        )

    # find super_cluster_id from name and perform case-insensitive match against super_cache
    super_id = None
    for sc_key_iter, name in artifacts.super_cache.items():
        parts = sc_key_iter.rsplit("__super", 1)
        if (
            parts[0] == internal_category and
            name.lower().strip() == super_cluster_name.lower().strip()
        ):
            super_id = int(parts[1])
            break

    if super_id is None:
        # build list of valid segment names for this category
        valid_names = [
            name
            for sc_key_iter, name in artifacts.super_cache.items()
            if sc_key_iter.startswith(internal_category)
        ]
        raise HTTPException(
            status_code  = 404,
            detail       = {
                "error"  : "Segment not found",
                "detail" : (f"No segment named '{super_cluster_name}' found in '{display_category}'. Valid segments: {valid_names}"),
                "path"   : f"/recommendations/{display_category}/{super_cluster_name}"
            }
        )

    sc_key = f"{internal_category}__super{super_id}"

    # check segment existence in reco cache
    if sc_key not in artifacts.reco_cache:
        raise HTTPException(
            status_code  = 404,
            detail       = {
                "error"  : "Recommendations not found",
                "detail" : (
                    f"Segment '{sc_key}' exists but has no cached "
                    f"recommendations. Ensure Section 9 of the pipeline "
                    f"completed successfully."
                ),
                "path"   : f"/recommendations/{display_category}/{super_cluster_name}"
            }
        )

    reco = artifacts.reco_cache[sc_key]

    # enrich with live sentiment distribution
    df       = artifacts.df_final_clean
    seg_df   = df[(df["category"] == internal_category) & (df["super_cluster"] == super_id)]

    sentiment_dist = None
    total_reviews  = None

    if len(seg_df) > 0:
        sent_counts = (
            seg_df["sentiment"]
            .value_counts(normalize=True)
            .mul(100).round(2)
        )
        sentiment_dist = {
            "positive" : float(sent_counts.get("positive", 0)),
            "negative" : float(sent_counts.get("negative", 0)),
            "neutral"  : float(sent_counts.get("neutral",  0))
        }
        total_reviews = int(len(seg_df))

    # build response
    return RecommendationResponse(
        sc_key                 = sc_key,
        category               = reco.get("category", internal_category),
        segment                = reco.get("segment", sc_key),
        overall_health         = reco.get("overall_health", "neutral"),
        total_reviews          = total_reviews,
        sentiment_distribution = sentiment_dist,
        recommendations        = reco.get("recommendations", []),
        opportunities          = reco.get("opportunities",  []),
        risk_flags             = reco.get("risk_flags",     [])
    )