# route that returns all categories with display names, segments, super cluster names & health status - populate dropdown menus or similar in frontend display

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from loader import ArtifactRegistry, get_artifacts
from config import (
    CATEGORIES,
    CATEGORY_DISPLAY_NAMES
)

router = APIRouter(
    prefix = "/categories",
    tags   = ["Categories"]
)

# response models
class SuperClusterInfo(BaseModel):
    super_cluster_id  : int
    super_cluster_name: str
    overall_health    : str
    total_reviews     : int
    sentiment_dist    : dict[str, float]

class CategoryInfo(BaseModel):
    internal_name : str
    display_name  : str
    total_reviews : int
    super_clusters: list[SuperClusterInfo]

class CategoriesResponse(BaseModel):
    total_categories: int
    categories      : list[CategoryInfo]

# route
@router.get(
    "",
    response_model = CategoriesResponse,
    summary        = "Get all categories with segments and super cluster names",
    description    = """
                        Returns all 7 categories with:
                        - Human-readable display names for UI dropdowns
                        - All super clusters per category with their LLM-assigned names
                        - Overall health per super cluster
                        - Review counts and sentiment distributions

                        Use this endpoint to populate category and segment dropdown menus.
                        No numerical IDs exposed to the user — everything is named.
                     """
)
async def get_categories(
    artifacts: ArtifactRegistry = Depends(get_artifacts)
):
    if not artifacts.is_ready:
        raise HTTPException(
            status_code  = 503,
            detail       = {
                "error"  : "Service unavailable",
                "detail" : "Artifacts not loaded. Check /health.",
                "path"   : "/categories"
            }
        )

    df   = artifacts.df_final_clean
    reco = artifacts.reco_cache

    categories_out = []

    for internal_name in CATEGORIES:
        display_name = CATEGORY_DISPLAY_NAMES.get(internal_name, internal_name)

        # filter dataframe to this category
        cat_df = df[df["category"] == internal_name]
        if len(cat_df) == 0:
            continue

        # get all super clusters for this category
        super_cluster_ids = sorted(cat_df["super_cluster"].unique().tolist())

        super_clusters_out = []
        for super_id in super_cluster_ids:
            sc_key   = f"{internal_name}__super{super_id}"
            sc_df    = cat_df[cat_df["super_cluster"] == super_id]
            sc_name  = artifacts.super_cache.get( sc_key, f"Segment {super_id}")     # super cluster name from cache
            seg_reco = reco.get(sc_key, {})     # health from recommendation cache
            health   = seg_reco.get("overall_health", "neutral")
            sent = (sc_df["sentiment"].value_counts(normalize=True).mul(100).round(2))  # sentiment distribution
            sentiment_dist = {
                "positive": float(sent.get("positive", 0)),
                "negative": float(sent.get("negative", 0)),
                "neutral" : float(sent.get("neutral",  0))
            }
            super_clusters_out.append(SuperClusterInfo(
                super_cluster_id   = int(super_id),
                super_cluster_name = sc_name,
                overall_health     = health,
                total_reviews      = int(len(sc_df)),
                sentiment_dist     = sentiment_dist
            ))
        categories_out.append(CategoryInfo(
            internal_name  = internal_name,
            display_name   = display_name,
            total_reviews  = int(len(cat_df)),
            super_clusters = super_clusters_out
        ))

    return CategoriesResponse(
        total_categories = len(categories_out),
        categories       = categories_out
    )