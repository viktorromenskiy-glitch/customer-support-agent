"""
Ticket triage: CSV in -> same CSV out, with category, priority and a "pass to a human" flag.

Usage (PowerShell, from the project folder that holds the two .joblib model files):
    python triage.py tickets.csv --dry-run      # checks the file, no API calls, no cost
    python triage.py tickets.csv                # writes tickets_triaged.csv next to the input
    python triage.py tickets.csv --min-category-confidence 0.8     # stricter: more tickets go to a human
    python triage.py tickets.csv --min-priority-confidence 0.6     # also use the priority confidence

A ticket goes to a human when the category confidence is below 0.70 (default). The priority is a
hint unless --min-priority-confidence is given. See the comment at MIN_CATEGORY_CONFIDENCE.

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

# Pass-to-human policy (chosen in plan step B3 from held-out ticket experiments):
# a ticket goes to a human when the category confidence (maximum class probability) is below
# MIN_CATEGORY_CONFIDENCE. The priority confidence is only a hint unless --min-priority-confidence is given.
# On held-out tickets (classifier trained on the first 12,000 of 20,000 tickets, labels set by a language
# model, not by humans) the share of wrong categories among tickets handled automatically was about
# 10% at 0.70 and about 6.5% at 0.80, with a cluster-bootstrap 95% interval of roughly 8-13% and 4.5-9%.
# This must be re-checked on the client's own tickets labelled by people (plan step B4).
MIN_CATEGORY_CONFIDENCE = 0.70
MIN_PRIORITY_CONFIDENCE = None    # None = priority confidence does not send a ticket to a human

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
# Columns of the progress file: raw model results only. The pass-to-human decision is made once,
# at the end, from the full-precision confidences, so every row follows the same thresholds
# even when a run is stopped and restarted with other thresholds.
PARTIAL_COLUMNS = ["row", "category", "category_confidence", "priority", "priority_confidence",
                   "empty_text", "model_version"]


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


def decide_human(category_conf, priority_conf, empty, min_category=MIN_CATEGORY_CONFIDENCE,
                 min_priority=MIN_PRIORITY_CONFIDENCE):
    """Returns (needs_human, reason). Reasons are joined with ';' when several apply.

    Args:
        category_conf: maximum class probability of the category classifier.
        priority_conf: maximum class probability of the priority classifier.
        empty: True when the ticket text is empty after cleaning (always goes to a human).
        min_category: below this category confidence the ticket goes to a human.
        min_priority: below this priority confidence the ticket goes to a human; None = not used.
    """
    if empty:
        return "yes", "empty_text"
    reasons = []
    if category_conf < min_category:
        reasons.append("low_category_confidence")
    if min_priority is not None and priority_conf < min_priority:
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


def check_threshold(name, value):
    """Stops with a clear message when a threshold is not a number between 0 and 1 (None is allowed for priority)."""
    if value is None:
        return
    if not (0.0 <= float(value) <= 1.0):
        raise SystemExit(f"{name} must be between 0 and 1, got {value}")


def run(input_path, output_path=None, models_dir=MODELS_DIR_DEFAULT, batch_size=64, dry_run=False, client=None,
        min_category=MIN_CATEGORY_CONFIDENCE, min_priority=MIN_PRIORITY_CONFIDENCE):
    """Runs the whole triage. Returns the absolute path of the output file (None for a dry run).

    Args:
        min_category: tickets with category confidence below this go to a human.
        min_priority: tickets with priority confidence below this go to a human; None = priority is only a hint.
    """
    check_threshold("--min-category-confidence", min_category)
    check_threshold("--min-priority-confidence", min_priority)
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
    print(f"Policy: to a human if category confidence < {min_category}"
          + (f" or priority confidence < {min_priority}" if min_priority is not None else " (priority is a hint only)"))
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
        if list(prev.columns) != PARTIAL_COLUMNS:
            raise SystemExit(f"Progress file {os.path.abspath(partial_path)} has an unexpected format "
                             f"(made by another version). Move it away and run again.")
        if len(prev) and set(prev["model_version"]) != {model_version}:
            raise SystemExit(f"Progress file {os.path.abspath(partial_path)} was made with other model files. "
                             f"Move it away and run again.")
        done = {int(r): True for r in prev["row"]}
        print(f"Resuming: {len(done)} of {n} rows already done in {partial_path}")
    else:
        pd.DataFrame(columns=PARTIAL_COLUMNS).to_csv(partial_path, index=False, encoding="utf-8-sig")

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
                out_rows.append({"row": i, "category": c, "category_confidence": f"{cc:.10f}", "priority": p,
                                 "priority_confidence": f"{pc:.10f}", "empty_text": "0",
                                 "model_version": model_version})
            else:
                out_rows.append({"row": i, "category": "", "category_confidence": "", "priority": "",
                                 "priority_confidence": "", "empty_text": "1", "model_version": model_version})
        pd.DataFrame(out_rows, columns=PARTIAL_COLUMNS).to_csv(partial_path, mode="a", header=False, index=False,
                                                               encoding="utf-8-sig")
        print(f"  {min(start + batch_size, len(pending))}/{len(pending)} rows done")

    # Merge and hard completeness check.
    part = pd.read_csv(partial_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    part["row"] = part["row"].astype(int)
    part = part.drop_duplicates("row", keep="last").set_index("row").sort_index()
    if len(part) != n or list(part.index) != list(range(n)):
        raise SystemExit(f"INCOMPLETE: {len(part)} of {n} rows have results. Run the same command again.")
    decisions = [decide_human(float(r.category_confidence) if r.category_confidence else 0.0,
                              float(r.priority_confidence) if r.priority_confidence else 0.0,
                              r.empty_text == "1", min_category, min_priority)
                 for r in part.itertuples()]
    part["needs_human"] = [d[0] for d in decisions]
    part["needs_human_reason"] = [d[1] for d in decisions]
    for col in ("category_confidence", "priority_confidence"):
        part[col] = [f"{float(v):.4f}" if v else "" for v in part[col]]
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
    ap.add_argument("--min-category-confidence", type=float, default=MIN_CATEGORY_CONFIDENCE,
                    help=f"tickets below this category confidence go to a human (default {MIN_CATEGORY_CONFIDENCE})")
    ap.add_argument("--min-priority-confidence", type=float, default=MIN_PRIORITY_CONFIDENCE,
                    help="also send tickets below this priority confidence to a human (default: not used)")
    a = ap.parse_args(argv)
    run(a.input, a.output, a.models_dir, a.batch_size, a.dry_run,
        min_category=a.min_category_confidence, min_priority=a.min_priority_confidence)


if __name__ == "__main__":
    main()
