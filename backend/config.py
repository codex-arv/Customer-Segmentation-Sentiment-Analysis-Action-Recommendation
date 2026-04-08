# configuration of all directories, paths, constants, variables & versions 

import re as _re
import os
from pathlib import Path

# version control
PIPELINE_VERSION = "v2"
LLM_VERSION      = "v6"

# base directories
BASE_ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", r"./artifacts"))
CS_BASE           = BASE_ARTIFACT_DIR / "customer_segmentation"
SENT_BASE         = BASE_ARTIFACT_DIR / "sentiment_classification"
LLM_BASE          = BASE_ARTIFACT_DIR / "llm"

# customer segmentation paths
TRANSFORMER_PATH       = CS_BASE / "transformer"       / "sentence_transformer.pkl"
UMAP_MODEL_PATH        = CS_BASE / "dim_red"           / "umap_model.pkl"
CLF_PATH               = CS_BASE / "classifier_model"  / "linearsvc.pkl"
BEST_CAT_CLF_PATH      = CS_BASE / "classifier_model"  / "best_cat_clf.pkl"
INFERENCE_SCALER_PATH  = CS_BASE / "classifier_model"  / "scaler_inference.pkl"
INFERENCE_PCA_PATH     = CS_BASE / "classifier_model"  / "pca_inference.pkl"
CLUSTER_TO_SUPER_PATH  = CS_BASE / "category_pipeline" / PIPELINE_VERSION / "cluster_to_super_mapping.json"
CLUSTER_CENTROIDS_PATH = CS_BASE / "classifier_model"  / "cluster_centroids.pkl"
VERSIONED_CATEGORY_DIR = CS_BASE / "category_pipeline" / PIPELINE_VERSION

# sentiment classification paths
SENT_CLF_PATH         = SENT_BASE / "embedding_based" / "best_sentiment_clf.pkl"
TFIDF_VECTORIZER_PATH = SENT_BASE / "vectors" / "tfidf_vectorizer.pkl"

# llm cache paths
LLM_TOPIC_PATH       = LLM_BASE / f"topic_names_{LLM_VERSION}.json"
LLM_SUPER_PATH       = LLM_BASE / f"super_cluster_names_{LLM_VERSION}.json"
LLM_RECO_PATH        = LLM_BASE / f"recommendations_{LLM_VERSION}.json"

# enriched dataset path
FINAL_ENRICHED_PATH  = (VERSIONED_CATEGORY_DIR / f"final_enriched_{PIPELINE_VERSION}_llm{LLM_VERSION}.csv")

# bertopic model paths
BERTOPIC_MODEL_PATHS = {
    "Beauty_and_Personal_Care__super0"   : VERSIONED_CATEGORY_DIR / "Beauty_and_Personal_Care"     / "super_0" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Beauty_and_Personal_Care__super1"   : VERSIONED_CATEGORY_DIR / "Beauty_and_Personal_Care"     / "super_1" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Clothing_Shoes_and_Jewelry__super0" : VERSIONED_CATEGORY_DIR / "Clothing_Shoes_and_Jewelry"   / "super_0" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Clothing_Shoes_and_Jewelry__super1" : VERSIONED_CATEGORY_DIR / "Clothing_Shoes_and_Jewelry"   / "super_1" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Electronics__super0"                : VERSIONED_CATEGORY_DIR / "Electronics"                  / "super_0" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Electronics__super1"                : VERSIONED_CATEGORY_DIR / "Electronics"                  / "super_1" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Grocery_and_Gourmet_Food__super1"   : VERSIONED_CATEGORY_DIR / "Grocery_and_Gourmet_Food"     / "super_1" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Home_and_Kitchen__super0"           : VERSIONED_CATEGORY_DIR / "Home_and_Kitchen"             / "super_0" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Home_and_Kitchen__super1"           : VERSIONED_CATEGORY_DIR / "Home_and_Kitchen"             / "super_1" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Home_and_Kitchen__super2"           : VERSIONED_CATEGORY_DIR / "Home_and_Kitchen"             / "super_2" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Sports_and_Outdoors__super0"        : VERSIONED_CATEGORY_DIR / "Sports_and_Outdoors"          / "super_0" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Sports_and_Outdoors__super1"        : VERSIONED_CATEGORY_DIR / "Sports_and_Outdoors"          / "super_1" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Tools_and_Home_Improvement__super0" : VERSIONED_CATEGORY_DIR / "Tools_and_Home_Improvement"   / "super_0" / "bertopic" / f"model_{PIPELINE_VERSION}",
    "Tools_and_Home_Improvement__super1" : VERSIONED_CATEGORY_DIR / "Tools_and_Home_Improvement"   / "super_1" / "bertopic" / f"model_{PIPELINE_VERSION}",
}

# all categories 
CATEGORIES = [
    "Beauty_and_Personal_Care",
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Grocery_and_Gourmet_Food",
    "Home_and_Kitchen",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
]

# display names of categories for the frontend
CATEGORY_DISPLAY_NAMES = {
    "Beauty_and_Personal_Care"   : "Beauty & Personal Care",
    "Clothing_Shoes_and_Jewelry" : "Clothing, Shoes & Jewellery",
    "Electronics"                : "Electronics",
    "Grocery_and_Gourmet_Food"   : "Grocery & Gourmet Food",
    "Home_and_Kitchen"           : "Home & Kitchen",
    "Sports_and_Outdoors"        : "Sports & Outdoors",
    "Tools_and_Home_Improvement" : "Tools & Home Improvement",
}

# reverse mapping to convert user input to internal format
DISPLAY_TO_CATEGORY = {v: k for k, v in CATEGORY_DISPLAY_NAMES.items()}

# order, for matching training scaler 
CATEGORY_COLUMNS = [
    "category_Beauty_and_Personal_Care",
    "category_Clothing_Shoes_and_Jewelry",
    "category_Electronics",
    "category_Grocery_and_Gourmet_Food",
    "category_Home_and_Kitchen",
    "category_Sports_and_Outdoors",
    "category_Tools_and_Home_Improvement",
]

# llm configuration
LLM_MODEL_PRIMARY   = "llama-3.3-70b-versatile"
LLM_MODEL_SECONDARY = "llama-3.1-8b-instant"
LLM_MODEL_TERTIARY  = "moonshotai/kimi-k2-instruct"
LLM_MODEL_DEFAULT   = LLM_MODEL_SECONDARY  

GROQ_API_BASE       = "https://api.groq.com/openai/v1"
LLM_TEMPERATURE     = 0
LLM_MAX_TOKENS      = 500
MAX_RETRIES         = 3
RETRY_DELAY_SECONDS = 5

# api configuration
API_TITLE       = "Customer Segmentation & Sentiment Analysis API"
API_DESCRIPTION = ("ML + NLP + LLM pipeline for customer review segmentation, topic extraction, sentiment prediction, and action recommendations.")
API_VERSION     = "1.0.0"

# cors middleware
FRONTEND_URL    = "http://localhost:5173"  # Vite default port
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000", 
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000"
]

# category name normalization function
def normalize_category(input_name: str) -> str:
    name = input_name.strip()

    if name in CATEGORIES:
        return name

    if name in DISPLAY_TO_CATEGORY:
        return DISPLAY_TO_CATEGORY[name]

    normalized = name
    normalized = _re.sub(r"\s*&\s*", " and ", normalized)
    normalized = _re.sub(r"\s+", "_", normalized)

    if normalized in CATEGORIES:
        return normalized

    normalized_lower = normalized.lower()
    for cat in CATEGORIES:
        if cat.lower() == normalized_lower:
            return cat

    for cat in CATEGORIES:
        if cat.lower().startswith(normalized_lower) or \
           normalized_lower.startswith(cat.lower()):
            return cat

    return input_name


# function to validate all critical artifacts on startup
def validate_paths():
    critical_paths = {
        "SentenceTransformer"   : TRANSFORMER_PATH,
        "UMAP model"            : UMAP_MODEL_PATH,
        "Cluster predictor"     : CLF_PATH,
        "Cluster centroids"     : CLUSTER_CENTROIDS_PATH,
        "Category classifier"   : BEST_CAT_CLF_PATH,
        "Inference scaler"      : INFERENCE_SCALER_PATH,
        "Inference PCA"         : INFERENCE_PCA_PATH,
        "Cluster-to-super map"  : CLUSTER_TO_SUPER_PATH,
        "Sentiment classifier"  : SENT_CLF_PATH,
        "TF-IDF vectorizer"     : TFIDF_VECTORIZER_PATH,
        "Topic names cache"     : LLM_TOPIC_PATH,
        "Recommendations cache" : LLM_RECO_PATH,
        "Enriched dataset"      : FINAL_ENRICHED_PATH,
    }

    missing = []
    for name, path in critical_paths.items():
        if not Path(path).exists():
            missing.append(f"  {name}: {path}")

    # check bertopic models
    for sc_key, path in BERTOPIC_MODEL_PATHS.items():
        if not Path(path).exists():
            missing.append(f"  BERTopic [{sc_key}]: {path}")

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} artifact(s) on startup:\n"
            + "\n".join(missing)
            + "\n\nEnsure unified_pipeline.ipynb has been run "
              "to completion before starting the backend."
        )

    return True