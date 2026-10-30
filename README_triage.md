# Ticket triage (CSV in, CSV out)

`triage.py` reads a CSV of support tickets and writes the same CSV with extra columns: a **category**, a **priority** and a **pass-to-human flag**. It embeds each ticket text (Voyage `voyage-4-lite`, a paid API call, no language-model calls) and runs two small trained classifiers saved in this folder.

## What you get

For each row, appended to your own columns:

| Column | Meaning |
|---|---|
| `category` | one of 8 fixed categories (Delivery / order status, Billing / refund, Account access, Quality complaint / compensation, Technical issue / bug, Service disruption, Positive feedback / thanks, General inquiry / other) |
| `category_confidence` | the classifier's top probability, 4 decimals |
| `priority` | Critical / High / Medium / Low |
| `priority_confidence` | same for priority |
| `needs_human` | `yes` or `no` |
| `needs_human_reason` | why `yes`: `empty_text`, `low_category_confidence`, `low_priority_confidence`, `text_too_long` (several reasons are joined with `;`) |
| `model_version` | identifies the model files and settings used for the row |

A ticket goes to a human when the category confidence is below **0.70** (strictly below), when the text is empty after cleaning, and, only if you switch these on, when the priority confidence is low or the text is long. The priority is a hint by default.

## Run it

```
pip install -r requirements-triage.txt
python triage.py sample_tickets.csv --dry-run     # checks the file, prints the size, no API calls, no cost
set VOYAGE_API_KEY=...                            # PowerShell: $env:VOYAGE_API_KEY="..."   Linux/macOS: export VOYAGE_API_KEY=...
python triage.py sample_tickets.csv               # writes sample_tickets_triaged.csv
```

Input: a `text` column is required; `id` and `company` are optional (`company` helps the model; an unknown company is allowed); any other columns are kept. The separator (comma, semicolon, tab) is detected; use `--encoding cp1251` for Excel files from some locales.

Useful options: `--output`, `--min-category-confidence 0.8` (stricter, more tickets go to a human), `--min-priority-confidence`, `--max-auto-chars 1000`, `--batch-size`, `--overwrite`. Run `python triage.py --help`.

## Safety of long runs

- Progress is saved after every batch in `<output>.partial.csv`. If the run stops, run the same command again: finished rows are not embedded (or paid for) twice.
- A result is written only if it passes a final check (one result row per input row, valid flags). Otherwise the script stops with `INCOMPLETE` and an existing result file is left untouched.
- An existing result file is never replaced silently (`--overwrite` is explicit). Two runs on the same output are refused (`<output>.lock`; delete it if a run was killed).
- Changing the threshold after a run has finished means embedding (and paying for) the tickets again, because the progress file is removed at the end.

## Check on 200 tickets (what was measured)

This check was done once, by hand, on a public corpus. It is **not** an estimate for your ticket stream.

- **Sample:** 200 support tweets from the public corpus that were not among the 20,000 training tickets. A quota sample: 5 companies, 40 each, three length groups (texts up to 287 characters). Two people labelled it independently with the same category definitions (agreement 186 of 200, 93%, Cohen's kappa 0.915). The definitions were written by the author, and one of the two labellers is the author.
- **What the tool did:** accepted 93 tickets automatically (46.5%) and passed 107 to a human (2 of them empty).
- **Result:** among the 93 automatically accepted tickets the model disagreed with the manual label as follows (per labeller):

| Labeller | Disagreements | Share | 95% Clopper-Pearson interval |
|---|---|---|---|
| 1 | 15 of 93 | 16.1% | 9.3-25.2% |
| 2 | 11 of 93 | 11.8% | 6.1-20.2% |

The two figures belong to two labellers and are not independent estimates of the truth. The difference between them is four rows, for example replies like "thanks!" that one labeller put into "General inquiry / other" and the other into "Positive feedback / thanks" (as the model did). The intervals ignore the quota design; a calculation adjusted for it gave similar, equally wide intervals (15.8% and 13.6%).
- **Priority** is only a hint: exact agreement with manual labels is 57.6-58.1%, with a tendency to overestimate (67 overestimates against 16-17 underestimates).
- **Rare classes** cannot be assessed: "Service disruption" occurs once in all 200 tickets and not at all among the accepted ones.

What this does **not** show: accuracy on your data; fresh data (the tweet dates lie inside the training period); texts longer than 287 characters; any other language or company; a calibrated threshold (a confidence of 0.70 is not a 70% probability). The training labels were assigned by a language model, not by people. The 14 tickets on which the two labellers disagreed with each other were not arbitrated, so there is no single final labelling of the 200 tickets. The check was run on 21 Oct 2026 and repeated on 30 Oct 2026 with `triage.py` version 12 and the two classifier files in this folder (the 200 texts were embedded again): all 200 rows gave the same category, priority, confidences (4 decimals) and pass-to-human flag as the saved results.

**How to confirm on your own tickets:** you give 100-200 of your tickets, your staff label them by your category definitions without seeing the model's answers, and we count disagreements among the automatically accepted ones. For the claim "not above 15%" (one-sided 95% Clopper-Pearson) at most 8 disagreements are allowed at 100 accepted tickets, 15 at 150 and 21 at 200. That rule assumes a binomial model and does not confirm that the model suits any given sample.

## Known limits (please read)

- **Accepted risk, paid batch:** if the process dies, or the service's answer fails the checks, between the service's reply and the write of the progress file, that one batch (up to 64 tickets) is embedded and paid again on restart. At the list price of this model ($0.02 per million tokens, from a third-party price list not checked against the provider) one batch costs a fraction of a cent. Identical texts in different rows are paid separately.
- If `<output>.tmp` exists as a folder, the script stops with a Python error after the paid step; paid rows are kept and a restart does not pay again.
- The classifiers were trained on **English** customer-support tweets of five companies (Apple, Amazon, Uber, Spotify, Delta), 8 fixed categories, short texts (under 300 characters). **Accuracy on your data is not known** until it is measured on your own tickets labelled by people. Category and priority labels used for training came from a language model, not from people.
- Texts longer than 8000 characters are cut before embedding. No accuracy measurement exists for long texts; `--max-auto-chars` sends them to a human.
- A header with a line break inside quotes is not supported.
- A confidence is shown with 4 decimals, but the decision uses the exact value, so a ticket shown as `0.7000` can still go to a human.
- The pass-to-human threshold 0.70 was chosen on held-out tickets with model-made labels; it must be re-checked on tickets labelled by people (acceptance test on your data).

## Docker (optional)

```
docker build -f Dockerfile.triage -t ticket-triage .
docker run --rm -v "$PWD":/data ticket-triage tickets.csv --dry-run          # no key, no cost
docker run --rm -e VOYAGE_API_KEY=... -v "$PWD":/data ticket-triage tickets.csv   # real run, result next to the input
docker run --rm --entrypoint python ticket-triage /app/test_triage_policy.py /app   # offline tests
```

PowerShell: `${PWD}` instead of `"$PWD"`. The image holds only `triage.py`, the two classifiers and the tests (no knowledge base, no API server); the key is passed at run time. `Dockerfile.triage.dockerignore` limits the build context to those files. The older `Dockerfile` (API with knowledge base) is separate and unchanged.

## Tests (offline, no API key, no cost)

```
python test_triage_policy.py .
```

Run from this folder (the argument is the folder with the two `.joblib` files). 162 checks use a fake embedding client. They need the pinned `scikit-learn` version above to load the models.

## Files

`triage.py` (the tool), `test_triage_policy.py`, `category_classifier_embeddings.joblib` and `priority_classifier_embeddings.joblib` (the two classifiers), `sample_tickets.csv` (20 example tickets), `requirements-triage.txt`, `Dockerfile.triage` with `Dockerfile.triage.dockerignore`.
