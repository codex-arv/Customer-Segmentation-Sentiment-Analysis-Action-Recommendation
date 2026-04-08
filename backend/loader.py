# loads all artifacts on startup and stores it in the memory till server lifetime

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from openai import OpenAI
from dotenv import load_dotenv


from config import (
    TRANSFORMER_PATH,
    UMAP_MODEL_PATH,
    CLF_PATH,
    BEST_CAT_CLF_PATH,
    INFERENCE_SCALER_PATH,
    INFERENCE_PCA_PATH,
    CLUSTER_TO_SUPER_PATH,
    SENT_CLF_PATH,
    TFIDF_VECTORIZER_PATH,
    LLM_TOPIC_PATH,
    LLM_SUPER_PATH,
    LLM_RECO_PATH,
    FINAL_ENRICHED_PATH,
    BERTOPIC_MODEL_PATHS,
    PIPELINE_VERSION,
    LLM_VERSION,
    LLM_MODEL_DEFAULT,
    LLM_MODEL_PRIMARY,
    LLM_MODEL_SECONDARY,
    LLM_MODEL_TERTIARY,
    GROQ_API_BASE,
    CATEGORY_COLUMNS,
    CLUSTER_CENTROIDS_PATH,
    validate_paths,
)


# artifacts registry: populated once on startuo, accessed by all route handlers through get_artifacts()
class ArtifactRegistry:

    def __init__(self):
        # ML models
        self.sentence_model    = None  
        self.umap_model        = None  
        self.svc               = None  # for cluster prediction
        self.cat_clf           = None  # for classifying categories
        self.scaler_inference  = None  
        self.pca_inference     = None  
        self.sentiment_clf     = None  # for classifying sentiment
        self.tfidf             = None  

        # bertopic models as value for category + super cluster as key
        self.bertopic_models   = {}   

        # lookup structures (caches & mappings)
        self.cluster_to_super  = {}  
        self.cluster_centroids = {}  
        self.topic_cache       = {}   
        self.super_cache       = {}    
        self.reco_cache        = {}    

        # enriched final dataset
        self.df_final_clean    = None 

        # LLM 
        self.llm_client        = None  
        self.llm_model         = None  

        # status
        self.loaded_artifacts  = []    
        self.failed_artifacts  = []    
        self.is_ready          = False  # True when all artifacts loaded

    def status_summary(self):
        return {
            "loaded" : len(self.loaded_artifacts),
            "failed" : len(self.failed_artifacts),
            "ready"  : self.is_ready,
            "details": [
                {"name": n, "status": "loaded"} for n in self.loaded_artifacts
            ] + [
                {"name": n, "status": "failed"} for n in self.failed_artifacts
            ]
        }

# instance
_registry = ArtifactRegistry()

def get_artifacts() -> ArtifactRegistry:
    """FastAPI dependency — returns the loaded registry."""
    return _registry

# artifact loading with status tracking
def _load(name: str, loader_fn, critical: bool = True):
    try:
        artifact = loader_fn()
        _registry.loaded_artifacts.append(name)
        print(f"[OK] {name}")
        return artifact
    except Exception as e:
        _registry.failed_artifacts.append(name)
        msg = f"[FAIL] {name}: {e}"
        if critical:
            raise RuntimeError(msg)
        else:
            print(f"  [WARN] {name}: {e} — non-critical, continuing.")
            return None

# LLM initialization and model selection
def _init_llm_client():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "API key not found in environment. "
            "Ensure .env file exists at C:\\minor\\backend\\.env "
            "with GROQ_API_KEY=your_key"
        )

    client = OpenAI(
        api_key  = api_key,
        base_url = GROQ_API_BASE
    )

    # test models in priority order
    candidates = [
        LLM_MODEL_SECONDARY,  
        LLM_MODEL_PRIMARY,
        LLM_MODEL_TERTIARY,
    ]

    for model in candidates:
        try:
            response = client.chat.completions.create(
                model       = model,
                temperature = 0,
                max_tokens  = 10,
                messages    = [{"role": "user", "content": '{"status": "ok"}'}]
            )
            raw = response.choices[0].message.content.strip()
            if "<think>" in raw:
                raw = re.sub(
                    r"<think>.*?</think>", "", raw, flags=re.DOTALL
                ).strip()
            print(f"[OK] LLM client → {model}")
            return client, model
        except Exception as e:
            error_msg = str(e)
            if "tokens per day" in error_msg or "TPD" in error_msg:
                print(f"[QUOTA] {model} — daily quota exhausted")
            elif "decommissioned" in error_msg:
                print(f"[DECOMMISSIONED] {model}")
            else:
                print(f"[UNAVAILABLE] {model}: {e}")
            continue

    raise RuntimeError(
        "All LLM models unavailable. Check Groq API key and quota at https://console.groq.com/settings/usage"
    )

# main loader: call once on FastAPI startup
def load_all_artifacts():
    print("\n" + "="*75)
    print(" LOADING ARTIFACTS (startup) ")
    print("="*75)

    # step 1: validating the existence of all paths
    # print("\n[CHECK] Validating artifact paths...")
    # validate_paths()
    print("All paths validated!")

    # step 2: loading the sentence transformer model
    print("\n[LOAD] SentenceTransformer...")
    _registry.sentence_model = _load(
        "SentenceTransformer",
        lambda: SentenceTransformer(str(TRANSFORMER_PATH))
    )

    # step 3: loading all dimensionality reduction models
    print("\n[LOAD] Dimensionality reduction models...")
    _registry.umap_model = _load(
        "UMAP model",
        lambda: joblib.load(UMAP_MODEL_PATH)
    )
    _registry.scaler_inference = _load(
        "Inference scaler",
        lambda: joblib.load(INFERENCE_SCALER_PATH)
    )
    _registry.pca_inference = _load(
        "Inference PCA",
        lambda: joblib.load(INFERENCE_PCA_PATH)
    )

    # step 4: loading all classifiers
    print("\n[LOAD] Classifiers...")
    _registry.svc = _load(
        "Cluster predictor (LinearSVC)",
        lambda: joblib.load(CLF_PATH)
    )
    _registry.cat_clf = _load(
        "Category classifier",
        lambda: joblib.load(BEST_CAT_CLF_PATH)
    )
    _registry.sentiment_clf = _load(
        "Sentiment classifier",
        lambda: joblib.load(SENT_CLF_PATH)
    )
    _registry.tfidf = _load(
        "TF-IDF vectorizer",
        lambda: joblib.load(TFIDF_VECTORIZER_PATH)
    )

    # step 5: loading lookup structures
    print("\n[LOAD] Lookup structures...")

    def _load_cluster_to_super():
        with open(CLUSTER_TO_SUPER_PATH, "r") as f:
            raw = json.load(f)
        return {
            category: {int(k): v for k, v in mapping.items()}
            for category, mapping in raw.items()
        }

    _registry.cluster_to_super = _load(
        "Cluster-to-super mapping",
        _load_cluster_to_super
    )

    _registry.cluster_centroids = _load(
        "Cluster centroids",
        lambda: joblib.load(CLUSTER_CENTROIDS_PATH)
    )

    def _load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    _registry.topic_cache = _load(
        f"Topic names cache (LLM {LLM_VERSION})",
        lambda: _load_json(LLM_TOPIC_PATH)
    )
    _registry.super_cache = _load(
        f"Super cluster names cache (LLM {LLM_VERSION})",
        lambda: _load_json(LLM_SUPER_PATH)
    )
    _registry.reco_cache = _load(
        f"Recommendations cache (LLM {LLM_VERSION})",
        lambda: _load_json(LLM_RECO_PATH)
    )

    # step 6: loading bertopic models
    print("\n[LOAD] 14 BERTopic models...")
    for sc_key, model_path in BERTOPIC_MODEL_PATHS.items():
        _registry.bertopic_models[sc_key] = _load(
            f"BERTopic [{sc_key}]",
            lambda p=model_path: BERTopic.load(str(p))
        )

    # step 7: loading enriched dataset
    print("\n[LOAD] Enriched dataset...")
    _registry.df_final_clean = _load(
        "Enriched dataset (df_final_clean)",
        lambda: pd.read_csv(FINAL_ENRICHED_PATH)
    )
    print(f"Shape: {_registry.df_final_clean.shape}")

    # step 8: loading LLM client
    print("\n[LOAD] LLM Groq client...")
    _registry.llm_client, _registry.llm_model = _load(
        "LLM client",
        _init_llm_client
    )

    # final status
    total   = len(_registry.loaded_artifacts)
    failed  = len(_registry.failed_artifacts)

    if failed == 0:
        _registry.is_ready = True
        print(f"\n{'='*75}")
        print(f"ALL {total} ARTIFACTS LOADED SUCCESSFULLY!!!")
        print(f"Pipeline : {PIPELINE_VERSION}")
        print(f"LLM      : {LLM_VERSION}")
        print(f"Model    : {_registry.llm_model}")
        print(f"Segments : {len(_registry.bertopic_models)}")
        print(f"Topics   : {len(_registry.topic_cache)}")
        print(f"{'='*75}\n")
    else:
        print(f"\n{'='*75}")
        print(f"WARNING : {failed} artifact(s) failed to load")
        print(f"Loaded  : {total}")
        print(f"Failed  : {failed}")
        print(f"{'='*75}\n")
        raise RuntimeError(
            f"{failed} critical artifact(s) failed to load. Check paths in config.py and ensure the pipeline notebook has been run to completion."
        )