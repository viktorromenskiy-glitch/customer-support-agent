"""
Agent orchestrator: a single embedding per ticket -> category classification +
priority classification + semantic search -> decision (draft a response or
escalate to a human).

The embedding is computed ONCE per ticket and reused for all three downstream
tasks - earlier, classification ran on TF-IDF and retrieval made a separate
Voyage call; now everything runs on one embedding, which is both more accurate
and avoids a redundant API call.

Defensive note: category/priority predictions are remapped from Russian to
English right after prediction (no-op if already English) - a persistent
environment issue kept reverting label files between separate script runs,
so this guarantees correct behavior regardless of which classifier file
version ends up loaded.

Before running:
1. pip install voyageai anthropic joblib pandas numpy scikit-learn scipy
2. $env:VOYAGE_API_KEY="..."   $env:ANTHROPIC_API_KEY="..."   (same terminal)
3. Make sure these are present alongside this script: category_classifier_embeddings.joblib,
   priority_classifier_embeddings.joblib, kb_embeddings.npy, kb_with_index.csv
"""
import numpy as np
import pandas as pd
import joblib
import re
import time
import voyageai
from anthropic import Anthropic
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

VOYAGE_MODEL = "voyage-4-lite"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Escalation thresholds by priority - the higher the priority, the stricter
# the required retrieval confidence before the system trusts an automatic draft.
ESCALATION_THRESHOLDS = {
    "Critical": 0.70,
    "High": 0.55,
    "Medium": 0.40,
    "Low": 0.40,
}

CATEGORY_RU_TO_EN = {
    "Доставка / статус заказа": "Delivery / order status",
    "Оплата / возврат средств": "Billing / refund",
    "Доступ к аккаунту": "Account access",
    "Жалоба на качество / компенсация": "Quality complaint / compensation",
    "Техническая проблема / баг": "Technical issue / bug",
    "Сбой сервиса": "Service disruption",
    "Положительный отзыв / благодарность": "Positive feedback / thanks",
    "Общее обращение / прочее": "General inquiry / other",
}
PRIORITY_RU_TO_EN = {
    "Критический": "Critical",
    "Высокий": "High",
    "Средний": "Medium",
    "Низкий": "Low",
}

voyage_client = voyageai.Client()
claude_client = Anthropic()

print("Loading classifiers and knowledge base...")
category_bundle = joblib.load("category_classifier_embeddings.joblib")
priority_bundle = joblib.load("priority_classifier_embeddings.joblib")
kb_df = pd.read_csv("kb_with_index.csv")
kb_df["category_llm"] = kb_df["category_llm"].map(CATEGORY_RU_TO_EN).fillna(kb_df["category_llm"])
kb_vectors = np.load("kb_embeddings.npy")
print(f"Knowledge base: {len(kb_df):,} tickets ready for search")


def clean(text):
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub(r'https?://\S+', '', text)
    return text


def get_embedding(text):
    """Computes the ticket's embedding ONCE - reused for both classification and retrieval."""
    for attempt in range(5):
        try:
            result = voyage_client.embed([clean(text)], model=VOYAGE_MODEL, input_type="query")
            return np.array(result.embeddings)
        except Exception as e:
            wait = 25 * (attempt + 1)
            print(f"  Embedding error, attempt {attempt+1}: waiting {wait}s... ({str(e)[:80]})")
            time.sleep(wait)
    raise RuntimeError("Could not get an embedding after all attempts")


def build_features(embedding, bundle, company, is_dm_escalation):
    """Same feature assembly used at training time: embedding + company (OHE) + is_dm_escalation."""
    company_encoded = bundle["ohe"].transform(pd.DataFrame([{"company": company}]))
    return hstack([
        csr_matrix(embedding),
        company_encoded,
        csr_matrix(np.array([[is_dm_escalation]])),
    ])


def classify_from_embedding(embedding, company, is_dm_escalation):
    """Classifies category and priority from an already-computed embedding."""
    cat_features = build_features(embedding, category_bundle, company, is_dm_escalation)
    category = category_bundle["classifier"].predict(cat_features)[0]
    category = CATEGORY_RU_TO_EN.get(category, category)
    category_proba = category_bundle["classifier"].predict_proba(cat_features).max()

    pri_features = build_features(embedding, priority_bundle, company, is_dm_escalation)
    priority = priority_bundle["classifier"].predict(pri_features)[0]
    priority = PRIORITY_RU_TO_EN.get(priority, priority)
    priority_proba = priority_bundle["classifier"].predict_proba(pri_features).max()

    return category, category_proba, priority, priority_proba


def retrieve_similar_from_embedding(embedding, k=5):
    """Semantic search for similar past tickets, using an already-computed embedding."""
    sims = cosine_similarity(embedding, kb_vectors)[0]
    top_idx = np.argsort(-sims)[:k]
    retrieved = kb_df.iloc[top_idx].copy()
    retrieved["similarity"] = sims[top_idx]
    return retrieved


def draft_response(text, retrieved):
    """Generates a draft response grounded in the retrieved precedents."""
    examples = "\n\n".join(
        f"Similar past request: {r['request_text']}\nHow it was answered then: {r['response_text']}"
        for _, r in retrieved.head(3).iterrows()
    )
    system_prompt = (
        "You are a customer support assistant. Based on similar past requests and how "
        "they were answered, draft a short, polite response to the new customer message. "
        "Do not invent facts that aren't present in the examples - if unsure, suggest asking "
        "the customer for more details instead."
    )
    user_msg = f"Similar past examples:\n\n{examples}\n\nNew customer message:\n{text}\n\nDraft response:"
    resp = claude_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=200, temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text.strip()


def handle_ticket(text, company="AmazonHelp", is_dm_escalation=False, verbose=True):
    """Full cycle: one embedding -> classify -> retrieve -> decide -> respond or escalate."""
    embedding = get_embedding(text)  # the only Voyage call in the whole cycle

    category, cat_conf, priority, pri_conf = classify_from_embedding(embedding, company, is_dm_escalation)
    retrieved = retrieve_similar_from_embedding(embedding, k=5)
    top_similarity = retrieved.iloc[0]["similarity"]

    threshold = ESCALATION_THRESHOLDS[priority]
    should_escalate = top_similarity < threshold

    if verbose:
        print(f"\n{'='*70}")
        print(f"Ticket: {text[:100]}")
        print(f"Category: {category} (confidence: {cat_conf:.2f})")
        print(f"Priority: {priority} (confidence: {pri_conf:.2f})")
        print(f"Best match in knowledge base: {top_similarity:.3f} (threshold for this priority: {threshold:.2f})")

    if should_escalate:
        if verbose:
            print(f"-> DECISION: escalate to a human (priority \"{priority}\" requires a closer match)")
        return {"action": "escalate", "category": category, "priority": priority, "top_similarity": top_similarity}
    else:
        draft = draft_response(text, retrieved)
        if verbose:
            print(f"-> DECISION: draft response generated")
            print(f"Draft: {draft}")
        return {"action": "draft", "category": category, "priority": priority,
                "top_similarity": top_similarity, "draft": draft}


if __name__ == "__main__":
    df = pd.read_csv("archive/twitter_support_labeled_FINAL.csv")
    df["request_created_at"] = pd.to_datetime(df["request_created_at"], errors="coerce", utc=True)
    df = df.sort_values("request_created_at").reset_index(drop=True)
    test_examples = df.iloc[int(len(df)*0.8):].sample(5, random_state=7)

    for _, row in test_examples.iterrows():
        handle_ticket(row["request_text"], company=row["company"], is_dm_escalation=row["is_dm_escalation"])
        time.sleep(20)  # pause to respect the 3 requests/minute free-tier limit