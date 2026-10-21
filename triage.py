"""
Ticket triage: CSV in -> same CSV out, with category, priority and a "pass to a human" flag.

Usage (PowerShell, from the project folder that holds the two .joblib model files):
    python triage.py tickets.csv --dry-run      # checks the file, no API calls, no cost
    python triage.py tickets.csv                # writes tickets_triaged.csv next to the input
    python triage.py tickets.csv --overwrite    # replace an existing result file (embeds and pays again)
    python triage.py tickets.csv --min-category-confidence 0.8     # stricter: more tickets go to a human
    python triage.py tickets.csv --min-priority-confidence 0.6     # also use the priority confidence
    python triage.py tickets.csv --max-auto-chars 1000             # longer texts always go to a human

A ticket goes to a human when the category confidence is below 0.70 (default). The priority is a
hint unless --min-priority-confidence is given. Texts longer than --max-auto-chars (off by default) also go
to a human: the classifiers were trained on short tweets (under 300 characters) and no accuracy
measurement exists for long texts. See the comment at MIN_CATEGORY_CONFIDENCE.

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
- an existing result file is never replaced silently: a repeated command stops with a message (use
  --overwrite or another --output on purpose);
- two runs on the same output file at the same time are refused (<output>.lock; if a run was killed
  and the next one says the lock exists, delete that file);
- the absolute path of the result file is printed last.

Known limits (see README): the classifiers were trained on English customer-support tweets of
five companies, with eight fixed categories; accuracy on other data is not known.
Paid results of a batch are lost (and paid again) if the process dies, or the answer fails the checks, between
the service's reply and the progress write; identical texts in different rows are paid separately.
The output shows confidences with 4 decimals, but the decision uses the exact value, so a ticket shown
as 0.7000 can still be passed to a human. When a run is finished the progress file is deleted: running
again with another threshold embeds (and pays for) the tickets again. Texts longer than 8000 characters
are cut to that length before embedding (the number of such texts is printed).
"""
import argparse
import contextlib
import csv
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
EMBED_DIM = 1024          # size of one embedding; both classifiers must expect exactly this plus the company encoding
EMBED_INPUT_TYPE = "document"
LONG_TEXT_NOTICE_CHARS = 1000   # the run prints how many texts are longer than this (all training texts were under 300)
PREPROCESS_VERSION = "1"   # raise when clean() or the way texts are cut changes; part of model_version
MAX_TEXT_CHARS = 8000     # longer cleaned texts are cut to this length before embedding (training texts were short tweets)
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


def validate_partial(prev, path):
    """Stops when the progress file holds values this script could not have written (hand edits, damage)."""
    def damaged(why):
        raise SystemExit(f"Progress file {os.path.abspath(path)} is damaged ({why}). Move it away and run again "
                         f"(rows already paid for will be embedded again).")
    if not prev["row"].str.fullmatch(r"\d+").all():
        damaged("a row number is not a whole number")
    rows = pd.to_numeric(prev["row"])
    if rows.duplicated().any():
        damaged("the same row number appears more than once")
    if not prev["empty_text"].isin(["0", "1"]).all():
        damaged("empty_text must be 0 or 1")
    full = prev[prev["empty_text"] == "0"]
    for col in ("category_confidence", "priority_confidence"):
        v = pd.to_numeric(full[col], errors="coerce")
        if v.isna().any() or ((v < 0) | (v > 1)).any():
            damaged(f"{col} is not a number between 0 and 1")
    if (full["category"] == "").any() or (full["priority"] == "").any():
        damaged("an empty category or priority for a non-empty ticket")
    known_cat = set(CATEGORY_RU_TO_EN.values())
    known_pri = set(PRIORITY_RU_TO_EN.values())
    if not full["category"].isin(known_cat).all() or not full["priority"].isin(known_pri).all():
        damaged("a category or priority that this script does not know")
    blank = prev[prev["empty_text"] == "1"]
    if (blank[["category", "category_confidence", "priority", "priority_confidence"]] != "").any().any():
        damaged("results saved for a ticket that is empty")


def check_partial_against_input(prev, hashes, empty, path):
    """Every saved row must belong to this input: same row number, same text and company (fingerprint),
    and the same empty/non-empty status."""
    n = len(hashes)
    for r, h, e in zip(prev["row"], prev["text_hash"], prev["empty_text"]):
        i = int(r)
        if i >= n or hashes[i] != h:
            raise SystemExit(f"Progress file {os.path.abspath(path)} belongs to a different input file "
                             f"(row {r} does not match). Move it away or use another --output.")
        if (e == "1") != bool(empty.iloc[i]):
            raise SystemExit(f"Progress file {os.path.abspath(path)} is damaged (row {r}: empty_text does not match "
                             f"the input). Move it away and run again (rows already paid for will be embedded again).")


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
    version = (f"{VOYAGE_MODEL}+LR cat:{file_digest(paths[0])} pri:{file_digest(paths[1])} "
               f"in:{EMBED_INPUT_TYPE} cut:{MAX_TEXT_CHARS} prep:{PREPROCESS_VERSION}")
    return joblib.load(paths[0]), joblib.load(paths[1]), version


def check_models(cat_bundle, pri_bundle):
    """Checks, before anything is paid for, that both classifiers expect EMBED_DIM numbers plus their own
    company encoding. Stops with a clear message otherwise."""
    for name, b in (("category", cat_bundle), ("priority", pri_bundle)):
        if not isinstance(b, dict) or "ohe" not in b or "classifier" not in b:
            raise SystemExit(f"The {name} model file does not have the expected content (ohe and classifier).")
        dim = b["classifier"].n_features_in_ - sum(len(c) for c in b["ohe"].categories_)
        if dim != EMBED_DIM:
            raise SystemExit(f"The {name} model expects {dim} embedding numbers, this script produces {EMBED_DIM}. "
                             f"Wrong model file?")
        label_map = CATEGORY_RU_TO_EN if name == "category" else PRIORITY_RU_TO_EN
        labels = {label_map.get(c, c) for c in b["classifier"].classes_}
        if not labels <= set(label_map.values()):
            raise SystemExit(f"The {name} model knows classes this script does not: "
                             f"{sorted(labels - set(label_map.values()))}. Wrong model file?")


@contextlib.contextmanager
def run_lock(output_path):
    """One run at a time per output file: a second run would write into the same progress file."""
    path = output_path + ".lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"Another run is using {os.path.abspath(output_path)} (lock file {os.path.abspath(path)} exists). "
                         f"If no other run is active (for example the previous one was killed), delete the lock file "
                         f"and run again.")
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


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
                 min_priority=MIN_PRIORITY_CONFIDENCE, too_long=False):
    """Returns (needs_human, reason). Reasons are joined with ';' when several apply.

    Args:
        category_conf: maximum class probability of the category classifier.
        priority_conf: maximum class probability of the priority classifier.
        empty: True when the ticket text is empty after cleaning (always goes to a human).
        min_category: below this category confidence the ticket goes to a human.
        min_priority: below this priority confidence the ticket goes to a human; None = not used.
        too_long: True when the text is longer than --max-auto-chars (goes to a human).
    """
    if empty:
        return "yes", "empty_text"
    reasons = []
    if too_long:
        reasons.append("text_too_long")
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
    status = getattr(error, "http_status", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        pass
    return status not in NOT_RETRIED_STATUS


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
            if attempt == max_attempts:
                print(f"  embedding error, attempt {attempt}/{max_attempts}: giving up ({str(e)[:80]})")
                break
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
    # Count fields the way a CSV reader does (a separator inside quotes does not count). A separator that
    # makes a column called "text" wins; otherwise the one giving the most columns; on a tie the comma wins
    # (between ";" and tab, ";" wins).
    fields = {}
    for d in (",", ";", "\t"):
        try:
            fields[d] = next(csv.reader([header], delimiter=d))
        except (csv.Error, StopIteration):
            fields[d] = [header]
    has_text = {d: any(f.strip().lower() == "text" for f in fields[d]) for d in fields}
    return max(fields, key=lambda d: (has_text[d], len(fields[d]), d == ","))


def check_field_counts(path, encoding, sep):
    """Every record must have as many fields as the header. A line break inside a text without quotes, or a
    missing separator, would otherwise cut one ticket into several rows without any error."""
    csv.field_size_limit(10 ** 8)
    bad = []
    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f, delimiter=sep)
            header = next(reader, None)
            if header is None:
                return
            for rec in reader:
                if rec and len(rec) != len(header):
                    bad.append((reader.line_num, len(rec)))
    except UnicodeDecodeError:
        return   # read_input reports the encoding problem
    if bad:
        first = ", ".join(f"line {ln} has {k}" for ln, k in bad[:5])
        raise SystemExit(f"{len(bad)} row(s) do not have {len(header)} fields like the header ({first}). Usually a "
                         f"line break inside a text that is not in quotes, or a missing or extra separator. "
                         f"Fix the file and run again.")


def read_input(path, encoding="utf-8-sig"):
    """Reads the input CSV. Returns (dataframe, separator, text_column, company_column, id_column).

    Column names are matched ignoring spaces and letter case ("Text", " company " work); the input columns
    keep their own names in the output. company_column / id_column are None when absent."""
    sep = detect_delimiter(path, encoding)
    check_field_counts(path, encoding, sep)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding, sep=sep)
    except UnicodeDecodeError:
        raise SystemExit(f"The input file is not readable as {encoding}. Save it as UTF-8 or pass --encoding.")
    except pd.errors.EmptyDataError:
        raise SystemExit("The input file is empty (no header line).")
    except pd.errors.ParserError as e:
        raise SystemExit(f"The input file cannot be read as a table: {str(e)[:150]}")
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
        min_category=MIN_CATEGORY_CONFIDENCE, min_priority=MIN_PRIORITY_CONFIDENCE, encoding="utf-8-sig",
        overwrite=False, max_auto_chars=None):
    """Runs the whole triage. Returns the absolute path of the output file (None for a dry run).

    Args:
        min_category: tickets with category confidence below this go to a human.
        min_priority: tickets with priority confidence below this go to a human; None = priority is only a hint.
        encoding: encoding of the input file (the output is always written as UTF-8 with a BOM, which Excel opens).
        overwrite: allow replacing an existing result file (the tickets are embedded and paid for again).
        max_auto_chars: tickets whose cleaned text is longer than this go to a human; None = no length rule.
    """
    check_threshold("--min-category-confidence", min_category)
    check_threshold("--min-priority-confidence", min_priority, allow_none=True)
    if max_auto_chars is not None and max_auto_chars < 1:
        raise SystemExit(f"--max-auto-chars must be at least 1, got {max_auto_chars}")
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
    empty = ~cleaned.str.contains(r"\w", regex=True)   # nothing but spaces/punctuation counts as empty
    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + "_triaged.csv"
    partial_path = output_path + ".partial.csv"
    if os.path.exists(output_path) and not os.path.exists(partial_path) and not overwrite:
        raise SystemExit(f"The result file already exists: {os.path.abspath(output_path)}. Nothing was done. Use another "
                         f"--output, or add --overwrite to replace it (the tickets are embedded and paid for again).")

    print(f"Input: {os.path.abspath(input_path)}")
    print(f"Rows: {n}  (empty after cleaning: {int(empty.sum())}); separator {sep!r}, encoding {encoding}")
    print("Company column: " + (f"'{company_col}'" if company_col else f"none, every row gets company '{DEFAULT_COMPANY}'"))
    print(f"Policy: to a human if category confidence < {min_category}"
          + (f" or priority confidence < {min_priority}" if min_priority is not None else " (priority is a hint only)")
          + (f"; or text longer than {max_auto_chars} characters" if max_auto_chars is not None else ""))
    notice_long = int((cleaned.str.len() > LONG_TEXT_NOTICE_CHARS).sum())
    if notice_long:
        print(f"NOTE: {notice_long} text(s) are longer than {LONG_TEXT_NOTICE_CHARS} characters. The classifiers were trained on "
              f"texts under 300 characters; their accuracy on longer texts is NOT verified. "
              f"Use --max-auto-chars N to send longer texts to a human.")
    to_embed_total = int((~empty).sum())
    too_long = int((cleaned.str.len() > MAX_TEXT_CHARS).sum())
    if too_long:
        print(f"NOTE: {too_long} text(s) are longer than {MAX_TEXT_CHARS} characters; only the first "
              f"{MAX_TEXT_CHARS} characters are embedded.")
    est_tokens = int(cleaned[~empty].str.slice(0, MAX_TEXT_CHARS).str.len().sum() / 4) + to_embed_total
    print(f"Rows to embed: {to_embed_total}; rough size estimate: ~{est_tokens} tokens "
          f"(character count / 4, an ESTIMATE, not a measurement)")
    if dry_run:
        print("Dry run: nothing was sent, no cost.")
        return None
    cat_bundle, pri_bundle, model_version = load_models(models_dir)
    check_models(cat_bundle, pri_bundle)
    with run_lock(output_path):
        companies = (df[company_col] if company_col else pd.Series([DEFAULT_COMPANY] * n))
        companies = companies.str.strip().replace("", DEFAULT_COMPANY)
        hashes = [row_hash(cleaned.iloc[i], companies.iloc[i]) for i in range(n)]
        known_companies = set(cat_bundle["ohe"].categories_[0])
        unseen = sorted(set(companies) - known_companies)
        if unseen:
            cnt = int((~companies.isin(known_companies)).sum())
            print(f"NOTE: {cnt} row(s) have a company the model has not seen {unseen[:5]}: "
                  f"the company part of their features is zero (the text still decides).")

        done = {}
        if os.path.exists(partial_path):
            prev = read_partial(partial_path)
            if list(prev.columns) != PARTIAL_COLUMNS:
                raise SystemExit(f"Progress file {os.path.abspath(partial_path)} has an unexpected format "
                                 f"(made by another version). Move it away and run again.")
            validate_partial(prev, partial_path)
            if len(prev) and set(prev["model_version"]) != {model_version}:
                raise SystemExit(f"Progress file {os.path.abspath(partial_path)} was made with other model files. "
                                 f"Move it away and run again.")
            check_partial_against_input(prev, hashes, empty, partial_path)
            done = {int(r): True for r in prev["row"]}
            print(f"Resuming: {len(done)} of {n} rows already done in {partial_path}")
        else:
            pd.DataFrame(columns=PARTIAL_COLUMNS).to_csv(partial_path, index=False, encoding="utf-8-sig")

        pending = [i for i in range(n) if i not in done]
        if os.path.exists(output_path) and not overwrite and os.path.exists(partial_path):
            if not pending:
                raise SystemExit(f"A finished result ({os.path.abspath(output_path)}) and a complete progress file both exist: "
                                 f"the previous run probably stopped at the very end. Nothing was done. Add --overwrite to "
                                 f"rebuild the result from the saved progress (no new cost), or delete the progress file.")
            print(f"NOTE: the existing file {os.path.abspath(output_path)} will be replaced when this run finishes.")
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
                emb, tok = embed_batch(client, [cleaned.iloc[i][:MAX_TEXT_CHARS] for i in live])
                tokens_used += tok
                if emb.shape != (len(live), EMBED_DIM):
                    raise RuntimeError(f"The service returned embeddings of shape {emb.shape}, expected "
                                       f"({len(live)}, {EMBED_DIM}). Nothing from this batch was saved.")
                norms = np.linalg.norm(emb, axis=1)
                if not np.allclose(norms, 1.0, atol=1e-3):
                    raise RuntimeError("Embeddings do not match the training setup (unit length). "
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
        validate_partial(part, partial_path)
        if set(part["model_version"]) != {model_version}:
            raise SystemExit(f"Progress file {os.path.abspath(partial_path)} holds results of other model files or "
                             f"settings. Move it away and run again.")
        check_partial_against_input(part, hashes, empty, partial_path)
        part["row"] = part["row"].astype(int)
        part = part.drop_duplicates("row", keep="last").set_index("row").sort_index()
        if len(part) != n or list(part.index) != list(range(n)):
            raise SystemExit(f"INCOMPLETE: {len(part)} of {n} rows have results. Run the same command again.")
        lengths = cleaned.str.len().tolist()
        decisions = [decide_human(float(r.category_confidence) if r.category_confidence else 0.0,
                                  float(r.priority_confidence) if r.priority_confidence else 0.0,
                                  r.empty_text == "1", min_category, min_priority,
                                  too_long=max_auto_chars is not None and lengths[k] > max_auto_chars)
                     for k, r in enumerate(part.itertuples())]
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
    ap.add_argument("--max-auto-chars", type=int, default=None,
                    help="tickets whose cleaned text is longer than this go to a human (default: no length rule; "
                         "the classifiers were trained on texts under 300 characters)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing result file (the tickets are embedded and paid for again)")
    a = ap.parse_args(argv)
    run(a.input, a.output, a.models_dir, a.batch_size, a.dry_run,
        min_category=a.min_category_confidence, min_priority=a.min_priority_confidence, encoding=a.encoding, overwrite=a.overwrite,
        max_auto_chars=a.max_auto_chars)


if __name__ == "__main__":
    main()
