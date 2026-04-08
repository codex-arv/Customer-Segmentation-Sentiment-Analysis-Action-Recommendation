# segment route returns all 14 segments with health status, topic breakdown, sentiment distribution and review counts

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from models import SegmentsResponse, SegmentSummary, TopicSummary, ErrorResponse
from loader import ArtifactRegistry, get_artifacts
from config import CATEGORIES

router = APIRouter(
    prefix = "/segments",
    tags   = ["Segments"]
)

@router.get(
    "",
    response_model = SegmentsResponse,
    responses      = {503: {"model" : ErrorResponse, "description" : "Artifacts not loaded"},},
    summary        = "Get all segments with health and topic breakdown",
    description    = """
                        Returns all 14 category+super_cluster segments with:
                        - Overall health (positive / neutral / concerning)
                        - Sentiment distribution (positive %, negative %, neutral %)
                        - Topic breakdown with per-topic sentiment
                        - Total review counts

                        Optionally filter by category.
                     """
)

async def get_segments(
    category : Optional[str] = Query(
        default     = None,
        description = "Filter by category name. Leave empty for all.",
        examples    = ["Electronics", "Beauty_and_Personal_Care"]
    ),
    artifacts: ArtifactRegistry = Depends(get_artifacts)
):
    if not artifacts.is_ready:
        raise HTTPException(
            status_code  = 503,
            detail       = {
                "error"  : "Service unavailable",
                "detail" : "Artifacts not loaded. Check '/health'.",
                "path"   : "/segments"
            }
        )

    # validate category filter
    if category:
        from config import normalize_category
        category = normalize_category(category)
        if category not in CATEGORIES:
            raise HTTPException(
                status_code  = 422,
                detail       = {
                    "error"  : "Invalid category",
                    "detail" : f"'{category}' is not valid. Call GET /categories for valid names.",
                    "path"   : "/segments"
                }
            )

    df    = artifacts.df_final_clean
    reco  = artifacts.reco_cache
    segments = []

    for sc_key in sorted(artifacts.bertopic_models.keys()):
        parts    = sc_key.rsplit("__super", 1)
        cat      = parts[0]
        super_id = int(parts[1])

        # apply category filter
        if category and cat != category: continue

        # filter dataframe to this particular segment
        seg_df = df[(df["category"] == cat) & (df["super_cluster"] == super_id)]

        if len(seg_df) == 0: continue

        # segment-level sentiment distribution
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

        # super cluster name
        sc_name = seg_df["super_cluster_name"].iloc[0] if "super_cluster_name" in seg_df.columns else sc_key

        # overall health from recommendation cache
        seg_reco       = reco.get(sc_key, {})
        overall_health = seg_reco.get("overall_health", "neutral")
        reco_count     = len(seg_reco.get("recommendations", []))

        # per-topic breakdown
        topics_data = []
        if "topic_name" in seg_df.columns:
            topic_groups = seg_df.groupby(["topic", "topic_name"], dropna=True)
            for (topic_id, topic_name), t_df in topic_groups:
                t_sent = (
                    t_df["sentiment"]
                    .value_counts(normalize=True)
                    .mul(100).round(2)
                )
                topics_data.append(TopicSummary(
                    topic_id               = int(topic_id),
                    topic_name             = str(topic_name),
                    review_count           = int(len(t_df)),
                    sentiment_distribution = {
                        "positive": float(t_sent.get("positive", 0)),
                        "negative": float(t_sent.get("negative", 0)),
                        "neutral" : float(t_sent.get("neutral",  0))
                    }
                ))
            topics_data.sort(key=lambda x: x.review_count, reverse=True)

        segments.append(SegmentSummary(
            sc_key                 = sc_key,
            category               = cat,
            super_cluster_id       = super_id,
            super_cluster_name     = sc_name,
            overall_health         = overall_health,
            total_reviews          = int(len(seg_df)),
            recommendation_count   = reco_count,
            sentiment_distribution = sentiment_dist,
            topics                 = topics_data
        ))

    return SegmentsResponse(
        total_segments = len(segments),
        segments       = segments
    )