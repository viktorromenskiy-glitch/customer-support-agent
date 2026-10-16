"""
Ticket triage: CSV in -> same CSV out, with category, priority and a "pass to a human" flag.

Usage (PowerShell, from the project folder that holds the two .joblib model files):
    python triage.py tickets.csv --dry-run      # checks the file, no API calls, no cost
    python triage.py tickets.csv                # writes tickets_triaged.csv next to the input

Input CSV: a required `text` column; optional `id` and `company`; any other columns are kept.
Output CSV: all input columns plus
    category, category_confidence, priority, priority_confidence,
    needs_human (yes/no), needs_human_reason, model_version

Needs VOYAGE_API_KEY in the environment (only for the real run). No language-model calls:
this step only embeds the text and runs the two trained classifiers.

Safety of long runs:
- progress is saved after every batch in <output>.partial.csv; if the run stops, run the same
  command again and finished rows are not embedded (or paid for) twice;
- the run is only complete when the output file has one result row per input row, otherwise it
  stops with an error;
- the absolute path of the result file is printed last.

Known limits (see README): the classifiers were trained on English customer-support tweets of
five companies, with eight fixed categories; accuracy on other data is not known.
"""
import argparse
import hashlib
import os
import re
import sys
import time

import numpy as np
import pandas as pd

VOYAGE_MODEL = "voyage-4-lite"
# The classifiers were trained on embeddings made with input_type="document"
# (training/build_semantic_rag.py, training/embed_remaining_test_set.py), so the same type is used here.
EMBED_INPUT_TYPE = "document"
MODELS_DIR_DEFAULT = "."
CATEGORY_MODEL_FILE = "category_classifier_embeddings.joblib"
PRIORITY_MODEL_FILE = "priority_classifier_embeddings.joblib"

# PROVISIONAL thresholds: a ticket goes to a human when the classifier is less sure than this.
# They are placeholders until they are chosen on held-out data (plan step B3).
MIN_CATEGORY_CONFIDENCE = 0.50
MIN_PRIORITY_CONFIDENCE = 0.50

DEFAULT_COMPANY = "unknown"   # the encoder ignores unseen companies (all-zero company features)

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
PRIORITY_RU_TO_EN = {"Критический": "Critical", "Высокий": "High", "Средний": "Medium", "Низкий": "Low"}

OUT_COLUMNS = ["category", "category_confidence", "priority", "priority_confidence",
               "needs_human", "needs_human_reason", "model_version"]


def clean(text):
    """Same cleaning as at training time: remove @mentions and links."""
    text = re.sub(r"@\w+", "", str(text))
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


def file_digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def load_models(models_dir):
    """Loads both classifier bundles and returns (category_bundle, priority_bundle, model_version).

    Raises:
        FileNotFoundError: if one of the two .joblib files is missing in models_dir.
    """
    import joblib
    paths = [os.path.join(models_dir, CATEGORY_MODEL_FILE), os.path.join(models_dir, PRIORITY_MODEL_FILE)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model file not found: {os.path.abspath(p)}")
    version = f"{VOYAGE_MODEL}+LR cat:{file_digest(paths[0])} pri:{file_digest(paths[1])}"
    return joblib.load(paths[0]), joblib.load(paths[1]), version


def predict(embeddings, companies, bundle, ru_to_en):
    """Returns (labels, confidences) for a batch.

    Args:
        embeddings: array (n, dim) of unit-norm Voyage embeddings.
        companies: list of n company names (unseen names are ignored by the encoder).
        bundle: dict with "ohe" and "classifier" as saved by training/train_final_embedding_classifiers.py.
        ru_to_en: label map applied after prediction (no-op for labels already in English).
    """
    from scipy.sparse import hstack, csr_matrix
    company_part = bundle["ohe"].transform(pd.DataFrame({"company": companies}))
    X = hstack([csr_matrix(embeddings), company_part])
    proba = bundle["classifier"].predict_proba(X)
    classes = bundle["classifier"].classes_
    labels = [ru_to_en.get(classes[i], classes[i]) for i in proba.argmax(axis=1)]
    return labels, proba.max(axis=1)


def decide_human(category_conf, priority_conf, empty):
    """Returns (needs_human, reason). Reasons are joined with ';' when several apply."""
    if empty:
        return "yes", "empty_text"
    reasons = []
    if category_conf < MIN_CATEGORY_CONFIDENCE:
        reasons.append("low_category_confidence")
    if priority_conf < MIN_PRIORITY_CONFIDENCE:
        reasons.append("low_priority_confidence")
    return ("yes" if reasons else "no"), ";".join(reasons)


def embed_batch(client, texts, max_attempts=5):
    """Embeds one batch with retries. Returns (embeddings array, tokens used).

    Raises:
        RuntimeError: if all attempts fail (progress of earlier batches is already saved).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            res = client.embed(texts, model=VOYAGE_MODEL, input_type=EMBED_INPUT_TYPE)
            return np.array(res.embeddings, dtype=float), int(getattr(res, "total_tokens", 0) or 0)
        except Exception as e:  # network, rate limit, ...
            wait = 25 * attempt
            print(f"  embedding error, attempt {attempt}/{max_attempts}: waiting {wait}s ({str(e)[:80]})")
            time.sleep(wait)
    raise RuntimeError("Embedding failed after all attempts. Progress is saved; run the same command again.")


def read_input(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "text" not in df.columns:
        raise SystemExit(f"Input file must have a 'text' column. Found: {list(df.columns)}")
    clash = [c for c in OUT_COLUMNS if c in df.columns]
    if clash:
        raise SystemExit(f"Input already has columns that the output would overwrite: {clash}. Rename them first.")
    return df


def run(input_path, output_path=None, models_dir=MODELS_DIR_DEFAULT, batch_size=64, dry_run=False, client=None):
    """Runs the whole triage. Returns the absolute path of the output file (None for a dry run)."""
    df = read_input(input_path)
    n = len(df)
    cleaned = df["text"].map(clean)
    empty = cleaned.str.len() == 0
    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + "_triaged.csv"
    partial_path = output_path + ".partial.csv"

    print(f"Input: {os.path.abspath(input_path)}")
    print(f"Rows: {n}  (empty after cleaning: {int(empty.sum())})")
    to_embed_total = int((~empty).sum())
    est_tokens = int(cleaned[~empty].str.len().sum() / 4) + to_embed_total
    print(f"Rows to embed: {to_embed_total}; rough size estimate: ~{est_tokens} tokens "
          f"(character count / 4, an ESTIMATE, not a measurement)")
    if dry_run:
        print("Dry run: nothing was sent, no cost.")
        return None
    if n == 0:
        raise SystemExit("Input has no rows.")

    cat_bundle, pri_bundle, model_version = load_models(models_dir)
    companies = (df["company"] if "company" in df.columns else pd.Series([DEFAULT_COMPANY] * n)).replace("", DEFAULT_COMPANY)

    done = {}
    if os.path.exists(partial_path):
        prev = pd.read_csv(partial_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        done = {int(r): True for r in prev["row"]}
        print(f"Resuming: {len(done)} of {n} rows already done in {partial_path}")
    else:
        pd.DataFrame(columns=["row"] + OUT_COLUMNS).to_csv(partial_path, index=False, encoding="utf-8-sig")

    pending = [i for i in range(n) if i not in done]
    if client is None and any(not empty.iloc[i] for i in pending):
        if not os.environ.get("VOYAGE_API_KEY"):
            raise SystemExit('VOYAGE_API_KEY is not set in this PowerShell window. Set it first: $env:VOYAGE_API_KEY="..."')
        import voyageai
        client = voyageai.Client()   # reads VOYAGE_API_KEY from the environment

    tokens_used = 0
    for start in range(0, len(pending), batch_size):
        rows = pending[start:start + batch_size]
        live = [i for i in rows if not empty.iloc[i]]
        results = {}
        if live:
            emb, tok = embed_batch(client, [cleaned.iloc[i] for i in live])
            tokens_used += tok
            cats, cat_conf = predict(emb, [companies.iloc[i] for i in live], cat_bundle, CATEGORY_RU_TO_EN)
            pris, pri_conf = predict(emb, [companies.iloc[i] for i in live], pri_bundle, PRIORITY_RU_TO_EN)
            for k, i in enumerate(live):
                results[i] = (cats[k], float(cat_conf[k]), pris[k], float(pri_conf[k]))
        out_rows = []
        for i in rows:
            if i in results:
                c, cc, p, pc = results[i]
                nh, why = decide_human(cc, pc, False)
                out_rows.append({"row": i, "category": c, "category_confidence": f"{cc:.3f}", "priority": p,
                                 "priority_confidence": f"{pc:.3f}", "needs_human": nh,
                                 "needs_human_reason": why, "model_version": model_version})
            else:
                out_rows.append({"row": i, "category": "", "category_confidence": "", "priority": "",
                                 "priority_confidence": "", "needs_human": "yes",
                                 "needs_human_reason": "empty_text", "model_version": model_version})
        pd.DataFrame(out_rows).to_csv(partial_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        print(f"  {min(start + batch_size, len(pending))}/{len(pending)} rows done")

    # Merge and hard completeness check.
    part = pd.read_csv(partial_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    part["row"] = part["row"].astype(int)
    part = part.drop_duplicates("row", keep="last").set_index("row").sort_index()
    if len(part) != n or list(part.index) != list(range(n)):
        raise SystemExit(f"INCOMPLETE: {len(part)} of {n} rows have results. Run the same command again.")
    out = pd.concat([df.reset_index(drop=True), part[OUT_COLUMNS].reset_index(drop=True)], axis=1)
    tmp = output_path + ".tmp"
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, output_path)
    check = pd.read_csv(output_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(check) != n or (check["needs_human"] == "").any():
        raise SystemExit("INCOMPLETE: the written file failed the final check. Do not use it.")
    os.remove(partial_path)

    print("\nSummary:")
    print(check["needs_human"].value_counts().to_dict(), "(needs_human)")
    if tokens_used:
        print(f"Voyage tokens used in this run: {tokens_used} "
              f"({tokens_used / max(1, len(pending) - int(empty.iloc[pending].sum())):.1f} per embedded ticket)")
    print(f"Model version: {model_version}")
    print("=" * 70)
    print("RESULT FILE (absolute path):")
    print(os.path.abspath(output_path))
    print("=" * 70)
    return os.path.abspath(output_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Triage support tickets: category, priority, pass-to-human flag.")
    ap.add_argument("input", help="input CSV with a 'text' column")
    ap.add_argument("--output", help="output CSV (default: <input>_triaged.csv)")
    ap.add_argument("--models-dir", default=MODELS_DIR_DEFAULT, help="folder with the two .joblib files")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="check the file and print the size, no API calls")
    a = ap.parse_args(argv)
    run(a.input, a.output, a.models_dir, a.batch_size, a.dry_run)


if __name__ == "__main__":
    main()
