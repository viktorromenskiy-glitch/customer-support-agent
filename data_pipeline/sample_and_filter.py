"""
Step 2: apply empirical length thresholds, DM-escalation labeling, language
detection, and stratified sampling down to a manageable size.

Works on top of the already-produced twitter_support_filtered.csv (fast,
since it's already much smaller than the original 2.8M rows).

Note on language detection: langdetect is noticeably unreliable on very
short text (a real, confirmed issue during development - e.g. "Still no
response" was misclassified as Norwegian). Detecting language on the
combined request+response text instead of the request alone fixed ~96%
of these misclassifications, by giving the detector more signal to work with.
"""
import re
import time
import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42  # for reproducibility

INPUT_FILE = '../twitter_support_filtered.csv'
OUTPUT_FILE = '../archive/twitter_support_final.csv'
PER_COMPANY_SAMPLE = 4000  # target: ~5 companies x 4000 = up to 20,000 rows
BUFFER_PER_COMPANY = 9000  # oversample first, since language/length filtering will remove some


def safe_detect(text):
    try:
        return detect(text)
    except Exception:
        return 'unknown'


df = pd.read_csv(INPUT_FILE)
print(f"Loaded: {len(df):,} pairs")

# --- 1. Pre-stratify to a working buffer (avoids running language detection
#        on all 400K+ rows - only on what we'll plausibly keep) ---
buf = []
for company, group in df.groupby('company'):
    n = min(BUFFER_PER_COMPANY, len(group))
    buf.append(group.sample(n, random_state=42))
df = pd.concat(buf, ignore_index=True)
print(f"Buffer after pre-stratification: {len(df):,}")

# --- 2. Language detection on the COMBINED request+response text (more reliable
#        than the request alone, which is often too short for confident detection) ---
combined = df['request_text'].astype(str) + ' ' + df['response_text'].astype(str)
print("Detecting language (combined request+response text)...")
start = time.time()
df['lang'] = combined.apply(safe_detect)
print(f"Language detection took {(time.time()-start)/60:.1f} min")
print(f"\nLanguage distribution (top 10):\n{df['lang'].value_counts().head(10)}")

df = df[df['lang'] == 'en'].copy()
print(f"\nAfter language filter (English only): {len(df):,}")

# --- 3. DM-escalation flag (on the company's response) - BEFORE length filtering,
#        so a short but meaningful "Please DM us" isn't lost ---
dm_pattern = re.compile(
    r'\b(dm|direct message|private message)\b.{0,15}\b(us|me)\b|'
    r'send.{0,10}(message|dm)|'
    r'follow.{0,10}(dm|message)',
    re.IGNORECASE
)
df['is_dm_escalation'] = df['response_text'].str.contains(dm_pattern, na=False)
print(f"\nFlagged as DM escalation: {df['is_dm_escalation'].sum():,} "
      f"({df['is_dm_escalation'].mean()*100:.1f}%)")

print("\n--- 5 random DM-escalation examples (manual sanity check) ---")
for t in df[df['is_dm_escalation']]['response_text'].sample(min(5, df['is_dm_escalation'].sum()), random_state=42):
    print(f"  - {t[:120]}")

# --- 4. Text length: thresholds based on the REAL distribution, not guessed ---
df['request_len'] = df['request_text'].str.len()
df['response_len'] = df['response_text'].str.len()

REQUEST_MIN_LEN = 20
RESPONSE_MIN_LEN = 20

# DM-escalations are protected from the length cutoff even if short
low_content = (
    ((df['request_len'] < REQUEST_MIN_LEN) | (df['response_len'] < RESPONSE_MIN_LEN))
    & (~df['is_dm_escalation'])
)
print(f"\nLow-content pairs (shorter than {REQUEST_MIN_LEN}/{RESPONSE_MIN_LEN} chars, "
      f"not a DM escalation): {low_content.sum():,} ({low_content.mean()*100:.1f}%)")

df = df[~low_content].copy()
print(f"Remaining after length filter: {len(df):,}")

# --- 5. Final stratified sample per company ---
sampled = []
for company, group in df.groupby('company'):
    n = min(PER_COMPANY_SAMPLE, len(group))
    sampled.append(group.sample(n, random_state=42))
df_final = pd.concat(sampled, ignore_index=True)

cols_keep = ['company', 'request_text', 'response_text', 'request_created_at',
             'response_created_at', 'is_dm_escalation', 'request_len', 'response_len']
df_final = df_final[cols_keep]

print(f"\n=== FINAL ===")
print(df_final['company'].value_counts())
print(f"Total rows: {len(df_final):,}")

df_final.to_csv(OUTPUT_FILE, index=False)
import os
size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
print(f"\nSaved to {OUTPUT_FILE}, size: {size_mb:.1f} MB")
