"""
Semantic RAG retrieval via Voyage AI embeddings.
Compares against the TF-IDF version on the same metric (category-match rate
in top-K) and on the same real "problem" example (Amazon lockers / "I've got").

Before running:
1. pip install voyageai pandas scikit-learn numpy
2. $env:VOYAGE_API_KEY="your_key"  (same terminal window before running)
"""
import re
import time
import numpy as np
import pandas as pd
import voyageai
from sklearn.metrics.pairwise import cosine_similarity

client = voyageai.Client()  # picks up the key from VOYAGE_API_KEY automatically

MODEL = "voyage-4-lite"  # cheap model, the free tier easily covers this project
BATCH_SIZE = 100

df = pd.read_csv("../archive/twitter_support_labeled_FINAL.csv")
df["request_created_at"] = pd.to_datetime(df["request_created_at"], errors="coerce", utc=True)
df = df.sort_values("request_created_at").reset_index(drop=True)

split_idx = int(len(df) * 0.8)
kb_df = df.iloc[:split_idx].copy().reset_index(drop=True)
query_df = df.iloc[split_idx:].copy().reset_index(drop=True)
print(f"Knowledge base: {len(kb_df):,} | Test queries (sampling 500 for comparison): 500")


def clean(text):
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub(r'https?://\S+', '', text)
    return text


kb_df["text_clean"] = kb_df["request_text"].apply(clean)
test_sample = query_df.sample(500, random_state=42).reset_index(drop=True)
test_sample["text_clean"] = test_sample["request_text"].apply(clean)


def embed_batch(texts, input_type, label=""):
    all_vecs = []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    total_tokens = 0
    failed_batches = []
    failed_positions = []  # (start_idx, length) for placeholder batches
    embedding_dim = None  # determined from the first successful batch
    for b in range(n_batches):
        batch = texts[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        # Guard against empty strings - the Voyage API rejects them
        batch_safe = [t if t and t.strip() else "(empty)" for t in batch]
        success = False
        for attempt in range(5):
            try:
                result = client.embed(batch_safe, model=MODEL, input_type=input_type)
                all_vecs.extend(result.embeddings)
                total_tokens += result.total_tokens
                if embedding_dim is None:
                    embedding_dim = len(result.embeddings[0])
                success = True
                break
            except Exception as e:
                wait = 25 * (attempt + 1)  # 25, 50, 75, 100, 125s - generous margin under the rate limit
                print(f"  {label} batch {b}, attempt {attempt+1}: waiting {wait}s... ({str(e)[:80]})")
                time.sleep(wait)
        if not success:
            print(f"  {label} batch {b} failed after all attempts - filling with a placeholder")
            failed_positions.append((len(all_vecs), len(batch)))
            all_vecs.extend([None] * len(batch))
            failed_batches.append(b)
        if (b + 1) % 10 == 0:
            print(f"  {label}: {b+1}/{n_batches} batches (failed so far: {len(failed_batches)})")
        time.sleep(20)  # polite pause between batches, under the 3 RPM free-tier limit
    if embedding_dim is None:
        embedding_dim = 1024
    for start, length in failed_positions:
        for i in range(start, start + length):
            all_vecs[i] = [0.0] * embedding_dim
    if failed_batches:
        print(f"  Total failed batches: {len(failed_batches)} of {n_batches} (embedding dim: {embedding_dim})")
    return np.array(all_vecs), total_tokens


print("\nEmbedding the knowledge base...")
kb_vectors, tok1 = embed_batch(kb_df["text_clean"].tolist(), "document", "KB")

# Save right away, before the second stage - this was the longest part
np.save("../archive/kb_embeddings.npy", kb_vectors)
kb_df.to_csv("kb_with_index.csv", index=False)
print("Intermediate save done: kb_embeddings.npy (in case the next step fails)")

print("Embedding the test queries...")
query_vectors, tok2 = embed_batch(test_sample["text_clean"].tolist(), "query", "Query")

print(f"\nTotal tokens used: {tok1+tok2:,} (out of 200,000,000 free)")

sims = cosine_similarity(query_vectors, kb_vectors)

K = 5
top_k_idx = np.argsort(-sims, axis=1)[:, :K]

category_matches = []
for i, (_, row) in enumerate(test_sample.iterrows()):
    retrieved_categories = kb_df.iloc[top_k_idx[i]]["category_llm"].values
    match_rate = (retrieved_categories == row["category_llm"]).mean()
    category_matches.append(match_rate)

print(f"\n=== SEMANTIC search: category-match rate in top-{K} = {np.mean(category_matches)*100:.1f}% ===")
print(f"(TF-IDF was: 36.3%, random baseline: 12.5%)")

# The same real "problem" example - Amazon lockers
lockers_idx = test_sample[test_sample["request_text"].str.contains("lockers", case=False, na=False)].index
if len(lockers_idx) > 0:
    i = lockers_idx[0]
    print(f"\n=== Same example (Amazon lockers) ===")
    print(f"Query: {test_sample.iloc[i]['request_text'][:100]}")
    print(f"Category: {test_sample.iloc[i]['category_llm']}")
    for rank, kb_idx in enumerate(top_k_idx[i]):
        kb_row = kb_df.iloc[kb_idx]
        print(f"  {rank+1}. [{kb_row['category_llm']}] sim={sims[i, kb_idx]:.3f} | {kb_row['request_text'][:80]}")

print("\nDone. kb_embeddings.npy and kb_with_index.csv hold the full index for the next step.")
