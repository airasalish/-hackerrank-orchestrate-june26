# Multi-Modal Evidence Review — Solution

Verifies damage claims (car, laptop, package) using **Google Gemini vision** — 100% free tier, no credit card required.

## Setup

```bash
pip install -r code/requirements.txt
```

Get a **free** Google AI Studio API key (no credit card, no billing):
> https://aistudio.google.com/app/apikey

Create a `.env` file in the repo root:
```
GOOGLE_API_KEY=your_key_here
```

## Run on test claims

```bash
python code/main.py
# Produces dataset/output.csv
```

Options:
```
--dataset-dir   Path to dataset directory (default: dataset)
--input         Input CSV override (default: dataset/claims.csv)
--output        Output CSV path (default: dataset/output.csv)
--delay         Seconds between API calls (default: 4.0 — stays within free 15 RPM limit)
```

## Run evaluation on sample claims

```bash
python code/evaluation/main.py
# Runs on dataset/sample_claims.csv, prints per-field accuracy metrics
```

## Free tier limits (gemini-1.5-flash)

| Limit         | Value               |
|---------------|---------------------|
| Requests/min  | 15 RPM              |
| Tokens/min    | 1,000,000 TPM       |
| Requests/day  | 1,500 req/day       |
| Cost          | **$0**              |

The default `--delay 4.0` keeps throughput at ~15 RPM. 44 test claims take ~3 minutes.

## How it works

1. Reads each claim row from the input CSV
2. Loads relevant user history and evidence requirements
3. Encodes all submitted images as inline base64 parts
4. Sends a single Gemini API call per claim with all images + context
5. Uses JSON-mode (`response_mime_type="application/json"`) for structured output
6. Sanitizes and validates all output values against allowed lists
7. Enforces inter-field logic (e.g. `valid_image=false` → `evidence_standard_met=false`)
8. Writes results to output CSV

## Two-pass pipeline

The full workflow is:

```
python code/main.py                          # pass 1: generate predictions
python code/evaluation/auditor.py \
  --predictions dataset/output.csv           # pass 2: Antigravity-style audit + auto-correct
```

The auditor:
- Sends each prediction row + its images to Gemini for independent review
- Returns PASS / WARN / FAIL per row with specific issues
- Auto-applies corrections for clearly wrong fields (validated against allowed lists)
- Writes `evaluation/audit_report.csv` with per-row verdicts

## Prompt improvements included (from audit)

- Severity rubric (glass_shatter / absent part / non-functional → `high`)
- issue_type discriminators (`glass_shatter` vs `crack`, `water_damage` vs `stain`)
- Cross-image consistency check (flags mismatched vehicles/objects across images)
- Stricter `contradicted` vs `not_enough_information` rules
- Annotation/drawn-circle detection → `text_instruction_present`
