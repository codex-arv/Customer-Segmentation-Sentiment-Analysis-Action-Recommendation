# live inference prediction code: takes title + review as input, runs ML pipeline, calls LLM, returns structured prediction result

import re
import json
import time
import numpy as np
from openai import OpenAI
from config import (
    CATEGORY_COLUMNS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    LLM_MODEL_PRIMARY,
    LLM_MODEL_SECONDARY,
    LLM_MODEL_TERTIARY,
)
from loader import ArtifactRegistry

# text cleaning & preprocessing
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"<.*?>",   " ", text)
    text = re.sub(r"\s+",     " ", text).strip()
    return text

# LLM call with retry & feedbacks
def _call_llm(
    client     : OpenAI,
    model      : str,
    prompt     : str,
    max_tokens : int = LLM_MAX_TOKENS
    ) -> dict:
    candidates = [model, LLM_MODEL_PRIMARY, LLM_MODEL_TERTIARY]
    seen = set()
    candidates = [m for m in candidates if not (m in seen or seen.add(m))]
    for attempt in range(1, MAX_RETRIES + 1):
        for candidate in candidates:
            try:
                response = client.chat.completions.create(
                    model       = candidate,
                    temperature = LLM_TEMPERATURE,
                    max_tokens  = max_tokens,
                    messages    = [
                        {
                            "role"   : "system",
                            "content": (
                                "You are a precise JSON-returning assistant. "
                                "You ONLY return valid JSON. "
                                "You NEVER add explanations, comments, or markdown formatting. "
                                "You NEVER start your response with words like 'Here' or 'Sure'."
                                "Your entire response must be parseable by json.loads()."
                            )
                        },
                        {
                            "role"   : "user",
                            "content": prompt
                        }
                    ]
                )

                raw = response.choices[0].message.content.strip()

                # strip markdown fences
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$",        "", raw)
                    raw = raw.strip()

                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed

            except Exception as e:
                error_msg = str(e)
                if "tokens per day" in error_msg or "TPD" in error_msg:
                    continue
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

    return {}

# build inference feature vector (scaler + umap + pca)
def _build_inference_features(
    embedding     : np.ndarray,
    category      : str,
    review_length : int,
    title_length  : int,
    word_count    : int,
    artifacts     : ArtifactRegistry
) -> np.ndarray:
    structured_row = [review_length, title_length, word_count]
    for col in CATEGORY_COLUMNS:
        structured_row.append(1 if col == f"category_{category}" else 0)
    structured_arr = np.array(structured_row).reshape(1, -1)

    # Scale → PCA → UMAP
    structured_scaled = artifacts.scaler_inference.transform(structured_arr)
    X_combined_inf    = np.hstack([embedding, structured_scaled])
    X_pca_inf         = artifacts.pca_inference.transform(X_combined_inf)
    X_umap_inf        = artifacts.umap_model.transform(X_pca_inf)

    return X_umap_inf

# unseen cluster fallback (most common): maps to nearest cluster using absolute distance
def _resolve_cluster(
    cluster       : int,
    category      : str,
    X_umap_inf    : np.ndarray,
    artifacts     : ArtifactRegistry
) -> int:
    known = artifacts.cluster_to_super.get(category, {})
    if cluster in known:
        return cluster
    if not known:
        raise ValueError(
            f"No cluster mapping found for category: {category}"
        )
    centroids = artifacts.cluster_centroids
    nearest   = min(
        known.keys(),
        key=lambda c: np.linalg.norm(
            X_umap_inf[0] - np.array(centroids[c])
        ) if c in centroids else float("inf")
    )
    return nearest

# main inference function
def predict_review(
    title    : str,
    text     : str,
    artifacts: ArtifactRegistry
) -> dict:
    # step 1: preprocess
    review_text   = (str(title).strip() + " " + str(text).strip()).strip()
    clean         = clean_text(review_text)
    review_length = len(str(text))
    title_length  = len(str(title))
    word_count    = len(str(text).split())

    # step 2: apply embeddings to the review ONLY
    embedding = artifacts.sentence_model.encode([str(text)], normalize_embeddings = True, convert_to_numpy = True)  

    # step 3: predict category 
    category = str(artifacts.cat_clf.predict(embedding)[0])

    # step 4: build features + reduce via UMAP & PCA
    X_umap_inf = _build_inference_features(embedding, category, review_length, title_length, word_count, artifacts)

    # step 5: predict cluster
    cluster = int(artifacts.svc.predict(X_umap_inf)[0])

    # step 6: resolve cluster to super cluster mapping
    cluster       = _resolve_cluster(cluster, category, X_umap_inf, artifacts)
    super_cluster = int(artifacts.cluster_to_super[category][cluster])
    sc_key        = f"{category}__super{super_cluster}"

    # step 7: predict topics
    if sc_key not in artifacts.bertopic_models:
        raise ValueError(
            f"No BERTopic model found for segment: {sc_key}"
        )
    topic_model   = artifacts.bertopic_models[sc_key]
    topics, _     = topic_model.transform([str(text)], embeddings=embedding)
    topic         = int(topics[0])

    # step 8: build topic context (v imp)
    topic_context = (f"{category}__super{super_cluster}__topic{topic}")

    # step 9: predict sentiment
    clean_vec = artifacts.tfidf.transform([clean])
    sentiment = str(artifacts.sentiment_clf.predict(clean_vec)[0])

    # step 10: lookups from cache
    topic_info      = artifacts.topic_cache.get(topic_context, {})
    topic_name      = topic_info.get("topic_name",    "Unknown Topic")
    topic_summary   = topic_info.get("topic_summary", "")
    sc_name         = artifacts.super_cache.get(sc_key, "Unknown Segment")
    batch_reco      = artifacts.reco_cache.get(sc_key, {})
    batch_reco_text = ""
    segment_health  = "N/A"
    opportunities   = []

    if batch_reco:
        recs = batch_reco.get("recommendations", [])
        if recs:
            batch_reco_text = recs[0].get("action", "")
        segment_health  = batch_reco.get("overall_health", "N/A")
        opportunities   = batch_reco.get("opportunities", [])

    # step 11: LLM interpretation per-review
    llm_prompt = f"""You are an expert e-commerce customer experience analyst
                    working for a business intelligence team. Your role is to interpret
                    individual customer reviews, explain sentiment drivers, and generate
                    immediate actionable guidance for both the product team and customer
                    service team.

                    A customer has submitted the following review:
                    Title  : {title}
                    Review : {text}

                    Our ML pipeline has analysed this review and produced:
                    Category      : {category}
                    Segment       : {sc_name}
                    Topic         : {topic_name}
                    Topic Context : {topic_summary}
                    Sentiment     : {sentiment}
                    Segment Intel : {batch_reco_text}

                    Your task is to interpret this specific review and generate targeted
                    recommendations. Base your response STRICTLY and SOLELY on the review text and
                    ML predictions above.

                    STRICT RULES:
                    1. review_summary — one sentence, captures the core customer message.
                    2. sentiment_explanation — explain the sentiment using specific words or phrases from the review text as evidence.
                    3. business_recommendation — specific, tied to this review's content, actionable for the product or brand team.
                    4. cs_action — specific, immediate action for customer service to resolve or acknowledge this customer's experience.
                    5. priority — "high" if negative sentiment or product failure, "medium" if neutral or mixed, "low" if positive.
                    6. Return ONLY a valid JSON object. No markdown, no explanation, no text outside the JSON. Only return valid JSON object.

                    OUTPUT FORMAT:
                    {{
                    "review_summary"          : "<one or two sentences>",
                    "sentiment_explanation"   : "<cite specific words from the review>",
                    "business_recommendation" : "<specific action for the product or brand team>",
                    "cs_action"               : "<specific immediate action for customer service team>",
                    "priority"                : "high | medium | low"
                    }}"""

    llm_result = _call_llm(
        client     = artifacts.llm_client,
        model      = artifacts.llm_model,
        prompt     = llm_prompt,
        max_tokens = LLM_MAX_TOKENS
    )

    # fallback if LLM returned empty or failed
    if not llm_result:
        llm_result = {
            "review_summary"          : "Interpretation unavailable.",
            "sentiment_explanation"   : sentiment,
            "business_recommendation" : batch_reco_text or "Review manually.",
            "cs_action"               : "Manual review required.",
            "priority"                : ("high" if sentiment == "negative" else "medium")
        }

    # step 12: compile final result
    return {
        "input": {
            "title"                   : title,
            "text"                    : text
        },
        "predictions": {
            "category"                : category,
            "cluster"                 : cluster,
            "super_cluster"           : super_cluster,
            "super_cluster_name"      : sc_name,
            "topic"                   : topic,
            "topic_name"              : topic_name,
            "topic_summary"           : topic_summary,
            "topic_context"           : topic_context,
            "sentiment"               : sentiment
        },
        "llm_interpretation": {
            "review_summary"          : llm_result.get("review_summary", ""),
            "sentiment_explanation"   : llm_result.get("sentiment_explanation", ""),
            "business_recommendation" : llm_result.get("business_recommendation", ""),
            "cs_action"               : llm_result.get("cs_action", ""),
            "priority"                : llm_result.get("priority", "medium")
        },
        "segment_context": {
            "segment_health"          : segment_health,
            "batch_recommendation"    : batch_reco_text,
            "segment_opportunities"   : opportunities
        }
    }