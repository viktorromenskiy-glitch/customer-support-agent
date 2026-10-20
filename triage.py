"""
Ticket triage: CSV in -> same CSV out, with category, priority and a "pass to a human" flag.

Usage (PowerShell, from the project folder that holds the two .joblib model files):
    python triage.py tickets.csv --dry-run      # checks the file, no API calls, no cost
    python triage.py tickets.csv                # writes tickets_triaged.csv next to the input
    python triage.py tickets.csv --min-category-confidence 0.8     # stricter: more tickets go to a human
    python triage.py tickets.csv --min-priority-confidence 0.6     # also use the priority confidence

A ticket goes to a human when the category confidence is below 0.70 (default). The priority is a
hint unless --min-priority-confidence is given. See the comment at MIN_CATEGORY_CONFIDENCE.

Input CSV: a required `text` column; optional `id` and `company`; any other columns are kept. The separator
(comma, semicolon or tab) is detected; column names are matched ignoring spaces and letter case.
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
PARTIAL_COLUMNS = ["row", "text_hash", "category", "category_confidence", "priority", "priority_confidence",
                   "empty_text", "model_version"]


def clean(text):
    """Same cleaning as at training time: remove @mentions and links."""
    text = re.sub(r"@\w+", "", str(text))
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


def row_hash(cleaned_text, company):
    """Short fingerprint of what a result was computed from. Stored with every saved row, so a progress
    file can never be reused for a different input file (the same row number but another ticket)."""
    return hashlib.sha256(f"{company}\x1f{cleaned_text}".encode("utf-8")).hexdigest()[:12]


def read_partial(path):
    """Reads the progress file; stops with a clear message when it is damaged."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise SystemExit(f"Progress file {os.path.abspath(path)} is damaged ({str(e)[:80]}). "
                         f"Move it away and run again (rows already paid for will be embedded again).")


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


# Errors that repeating the same request cannot fix (bad request, bad key): stop at once, no waiting.
NOT_RETRIED_NAMES = {"AuthenticationError", "InvalidRequestError", "MalformedRequestError"}
NOT_RETRIED_STATUS = {400, 401, 403, 404, 422}


def is_retriable(error):
    """True for temporary problems (timeouts, rate limits, 5xx, network); False when a retry cannot help."""
    if type(error).__name__ in NOT_RETRIED_NAMES:
        return False
    return getattr(error, "http_status", None) not in NOT_RETRIED_STATUS


def embed_batch(client, texts, max_attempts=5):
    """Embeds one batch with retries. Returns (embeddings array, tokens used).

    Temporary errors are retried with growing waits; errors that a retry cannot fix stop the run at once.
    Note: after a timeout the service may already have processed (and charged) the request, so a retry
    can be charged again; this cannot be excluded from the client side.

    Raises:
        RuntimeError: if the error is not retriable or all attempts fail (progress of earlier batches is saved).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            res = client.embed(texts, model=VOYAGE_MODEL, input_type=EMBED_INPUT_TYPE)
            return np.array(res.embeddings, dtype=float), int(getattr(res, "total_tokens", 0) or 0)
        except Exception as e:  # network, rate limit, ...
            if not is_retriable(e):
                raise RuntimeError(f"The service rejected the request and a retry cannot help: {str(e)[:200]}. "
                                   f"Progress is saved; fix the cause and run the same command again.")
            wait = 25 * attempt
            print(f"  embedding error, attempt {attempt}/{max_attempts}: waiting {wait}s ({str(e)[:80]})")
            time.sleep(wait)
    raise RuntimeError("Embedding failed after all attempts. Progress is saved; run the same command again.")


def detect_delimiter(path, encoding):
    """Picks the column separator from the header line: comma, semicolon (Excel in many locales) or tab."""
    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            header = f.readline()
    except UnicodeDecodeError:
        raise SystemExit(f"The input file is not readable as {encoding}. Save it as UTF-8 (in Excel: Save As > "
                         f"CSV UTF-8) or pass --encoding (for example --encoding cp1251).")
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=lambda d: (counts[d], d == ","))
    return best if counts[best] > 0 else ","


def read_input(path, encoding="utf-8-sig"):
    """Reads the input CSV. Returns (dataframe, separator, text_column, company_column, id_column).

    Column names are matched ignoring spaces and letter case ("Text", " company " work); the input columns
    keep their own names in the output. company_column / id_column are None when absent."""
    sep = detect_delimiter(path, encoding)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding, sep=sep)
    except UnicodeDecodeError:
        raise SystemExit(f"The input file is not readable as {encoding}. Save it as UTF-8 or pass --encoding.")
    names = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in names:
            raise SystemExit(f"Two columns are called the same after ignoring spaces and case: '{names[key]}' and '{c}'.")
        names[key] = c
    if "text" not in names:
        raise SystemExit(f"Input file must have a 'text' column (separator detected: {sep!r}). Found: {list(df.columns)}")
    clash = [c for c in df.columns if str(c).strip().lower() in OUT_COLUMNS]
    if clash:
        raise SystemExit(f"Input already has columns that the output would overwrite: {clash}. Rename them first.")
    return df, sep, names["text"], names.get("company"), names.get("id")


def check_threshold(name, value, allow_none=False):
    """Stops with a clear message when a threshold is not a number between 0 and 1."""
    if value is None:
        if allow_none:
            return
        raise SystemExit(f"{name} is required (a number between 0 and 1)")
    if not (0.0 <= float(value) <= 1.0):
        raise SystemExit(f"{name} must be between 0 and 1, got {value}")


def run(input_path, output_path=None, models_dir=MODELS_DIR_DEFAULT, batch_size=64, dry_run=False, client=None,
        min_category=MIN_CATEGORY_CONFIDENCE, min_priority=MIN_PRIORITY_CONFIDENCE, encoding="utf-8-sig"):
    """Runs the whole triage. Returns the absolute path of the output file (None for a dry run).

    Args:
        min_category: tickets with category confidence below this go to a human.
        min_priority: tickets with priority confidence below this go to a human; None = priority is only a hint.
        encoding: encoding of the input file (the output is always written as UTF-8 with a BOM, which Excel opens).
    """
    check_threshold("--min-category-confidence", min_category)
    check_threshold("--min-priority-confidence", min_priority, allow_none=True)
    if batch_size < 1:
        raise SystemExit(f"--batch-size must be at least 1, got {batch_size}")
    df, sep, text_col, company_col, id_col = read_input(input_path, encoding)
    if output_path is not None and os.path.abspath(output_path) == os.path.abspath(input_path):
        raise SystemExit("--output must not be the input file (the input would be overwritten).")
    if len(df) == 0:
        raise SystemExit("Input has no rows.")
    if id_col is not None and df[id_col].duplicated().any():
        print(f"WARNING: {int(df[id_col].duplicated().sum())} duplicated values in the '{id_col}' column; "
              f"rows are processed by position, ids are copied as they are.")
    n = len(df)
    cleaned = df[text_col].map(clean)
    empty = cleaned.str.len() == 0
    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + "_triaged.csv"
    partial_path = output_path + ".partial.csv"

    print(f"Input: {os.path.abspath(input_path)}")
    print(f"Rows: {n}  (empty after cleaning: {int(empty.sum())}); separator {sep!r}, encoding {encoding}")
    print("Company column: " + (f"'{company_col}'" if company_col else f"none, every row gets company '{DEFAULT_COMPANY}'"))
    print(f"Policy: to a human if category confidence < {min_category}"
          + (f" or priority confidence < {min_priority}" if min_priority is not None else " (priority is a hint only)"))
    to_embed_total = int((~empty).sum())
    est_tokens = int(cleaned[~empty].str.len().sum() / 4) + to_embed_total
    print(f"Rows to embed: {to_embed_total}; rough size estimate: ~{est_tokens} tokens "
          f"(character count / 4, an ESTIMATE, not a measurement)")
    if dry_run:
        print("Dry run: nothing was sent, no cost.")
        return None
    cat_bundle, pri_bundle, model_version = load_models(models_dir)
    companies = (df[company_col] if company_col else pd.Series([DEFAULT_COMPANY] * n))
    companies = companies.str.strip().replace("", DEFAULT_COMPANY)
    hashes = [row_hash(cleaned.iloc[i], companies.iloc[i]) for i in range(n)]

    done = {}
    if os.path.exists(partial_path):
        prev = read_partial(partial_path)
        if list(prev.columns) != PARTIAL_COLUMNS:
            raise SystemExit(f"Progress file {os.path.abspath(partial_path)} has an unexpected format "
                             f"(made by another version). Move it away and run again.")
        if len(prev) and set(prev["model_version"]) != {model_version}:
            raise SystemExit(f"Progress file {os.path.abspath(partial_path)} was made with other model files. "
                             f"Move it away and run again.")
        for r, h in zip(prev["row"], prev["text_hash"]):
            if int(r) >= n or hashes[int(r)] != h:
                raise SystemExit(f"Progress file {os.path.abspath(partial_path)} belongs to a different input file "
                                 f"(row {r} does not match). Move it away or use another --output.")
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
            norms = np.linalg.norm(emb, axis=1)
            if emb.shape[1] != cat_bundle["classifier"].n_features_in_ - len(cat_bundle["ohe"].categories_[0]) \
                    or not np.allclose(norms, 1.0, atol=1e-3):
                raise RuntimeError("Embeddings do not match the training setup (size or unit length). "
                                   "Nothing from this batch was saved.")
            cats, cat_conf = predict(emb, [companies.iloc[i] for i in live], cat_bundle, CATEGORY_RU_TO_EN)
            pris, pri_conf = predict(emb, [companies.iloc[i] for i in live], pri_bundle, PRIORITY_RU_TO_EN)
            for k, i in enumerate(live):
                results[i] = (cats[k], float(cat_conf[k]), pris[k], float(pri_conf[k]))
        out_rows = []
        for i in rows:
            if i in results:
                c, cc, p, pc = results[i]
                if not (np.isfinite(cc) and np.isfinite(pc)):
                    raise RuntimeError(f"A classifier returned a non-finite confidence for row {i}. Nothing saved.")
                out_rows.append({"row": i, "text_hash": hashes[i], "category": c, "category_confidence": repr(cc),
                                 "priority": p, "priority_confidence": repr(pc), "empty_text": "0",
                                 "model_version": model_version})
            else:
                out_rows.append({"row": i, "text_hash": hashes[i], "category": "", "category_confidence": "",
                                 "priority": "", "priority_confidence": "", "empty_text": "1",
                                 "model_version": model_version})
        pd.DataFrame(out_rows, columns=PARTIAL_COLUMNS).to_csv(partial_path, mode="a", header=False, index=False,
                                                               encoding="utf-8-sig")
        print(f"  {min(start + batch_size, len(pending))}/{len(pending)} rows done")

    # Merge and hard completeness check.
    part = read_partial(partial_path)
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
    out.to_csv(tmp, index=False, encoding="utf-8-sig", sep=sep)
    os.replace(tmp, output_path)
    check = pd.read_csv(output_path, dtype=str, keep_default_na=False, encoding="utf-8-sig", sep=sep)
    full = ~empty.reset_index(drop=True)
    bad = (len(check) != n
           or not check["needs_human"].isin(["yes", "no"]).all()
           or (check.loc[full.values, ["category", "category_confidence", "priority", "priority_confidence"]] == "").any().any()
           or (check["model_version"] == "").any())
    if bad:
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
    ap.add_argument("--encoding", default="utf-8-sig",
                    help="encoding of the input file (default utf-8-sig; Excel files from some locales need cp1251)")
    a = ap.parse_args(argv)
    run(a.input, a.output, a.models_dir, a.batch_size, a.dry_run,
        min_category=a.min_category_confidence, min_priority=a.min_priority_confidence, encoding=a.encoding)


if __name__ == "__main__":
    main()
