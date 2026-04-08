
# pydantic models: request & response schemas for all endpoints

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# constrained field values (enum)
class SentimentEnum(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral  = "neutral"

class HealthEnum(str, Enum):
    positive   = "positive"
    neutral    = "neutral"
    concerning = "concerning"

class PriorityEnum(str, Enum):
    high   = "high"
    medium = "medium"
    low    = "low"

class ImpactEnum(str, Enum):
    revenue      = "revenue"
    retention    = "retention"
    satisfaction = "satisfaction"
    brand        = "brand"

class UrgencyEnum(str, Enum):
    immediate  = "immediate"
    short_term = "short_term"
    long_term  = "long_term"

# request for prediction endpoint (POST) "/predict"
class PredictRequest(BaseModel):
    title: str = Field(
        ...,
        min_length  = 1,
        max_length  = 100,
        description = "Title of the customer review",
        examples    = ["Terrible product, stopped working after 2 days"]
    )
    text: str = Field(
        ...,
        min_length  = 1,
        max_length  = 5000,
        description = "Body text of the customer review",
        examples    = ["I bought this electronic device and it completely stopped working after just 2 days. Very disappointed."]
    )

# response for prediction endpoint (POST) "/predict"
class MLPredictions(BaseModel):
    category           : str
    cluster            : int
    super_cluster      : int
    super_cluster_name : str
    topic              : int
    topic_name         : str
    topic_summary      : str
    topic_context      : str
    sentiment          : SentimentEnum

class LLMInterpretation(BaseModel):
    review_summary          : str
    sentiment_explanation   : str
    business_recommendation : str
    cs_action               : str
    priority                : PriorityEnum

class SegmentContext(BaseModel):
    segment_health        : str
    batch_recommendation  : str
    segment_opportunities : list[str]

class PredictResponse(BaseModel):
    input              : dict[str, str]
    predictions        : MLPredictions
    llm_interpretation : LLMInterpretation
    segment_context    : SegmentContext

# response for segments (GET) "/segments"

class TopicSummary(BaseModel):
    topic_id               : int
    topic_name             : str
    review_count           : int
    sentiment_distribution : dict[str, float]

class SegmentSummary(BaseModel):
    sc_key                 : str
    category               : str
    super_cluster_id       : int
    super_cluster_name     : str
    overall_health         : HealthEnum
    total_reviews          : int
    recommendation_count   : int
    sentiment_distribution : dict[str, float]
    topics                 : list[TopicSummary]

class SegmentsResponse(BaseModel):
    total_segments : int
    segments       : list[SegmentSummary]

# response for recommendations for a specific category & super cluster (GET)
class Recommendation(BaseModel):
    action    : str
    rationale : str
    topic     : Optional[str] = "General"
    priority  : PriorityEnum
    impact    : ImpactEnum
    urgency   : UrgencyEnum

class RecommendationResponse(BaseModel):
    sc_key                 : str
    category               : str
    segment                : str
    overall_health         : HealthEnum
    total_reviews          : Optional[int] = None
    sentiment_distribution : Optional[dict[str, float]] = None
    recommendations        : list[Recommendation]
    opportunities          : list[str]
    risk_flags             : list[str]

# response for health check (GET)
class ArtifactStatus(BaseModel):
    name   : str
    status : str
    path   : str

class HealthResponse(BaseModel):
    status           : str
    pipeline_version : str
    llm_version      : str
    llm_model        : str
    artifacts_loaded : int
    artifacts_total  : int
    categories       : list[str]
    total_segments   : int
    total_topics     : int
    artifact_details : list[ArtifactStatus]

# shared error response
class ErrorResponse(BaseModel):
    error  : str
    detail : Optional[str] = None
    path   : Optional[str] = None