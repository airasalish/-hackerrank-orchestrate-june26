# Operational Analysis — Multi-Modal Evidence Review

## Model & Cost
**Model:** `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API  
**Cost: $0.00** — Groq free tier (no credit card required)

Free tier limits (on-demand):
- 30 RPM (requests per minute)
- 500,000 TPD (tokens per day, rolling window)

## Call volume

| Dataset | Rows | API calls |
|---|---|---|
| sample_claims.csv | 20 | 20 |
| claims.csv | 44 | 44 |
| **Total** | **64** | **64** |

One API call per claim. Each call includes all images for that claim (base64-encoded, resized to 384px max) plus the full prompt. No multi-turn conversation.

## Image count

- Sample set: ~34 images (avg ~1.7 per claim × 20 claims)
- Test set: ~75 images (avg ~1.7 per claim × 44 claims)
- **Total images processed: ~109**

## Token usage (estimates)

Per claim:
- System prompt: ~450 tokens
- User text (conversation + history + requirements): ~300 tokens
- Images: ~1,500–3,500 tokens per image (Groq llama-4-scout vision encoding)
- Output JSON: ~200 tokens
- **Estimated per-claim total: ~3,000–8,000 tokens**

Full 44-claim run:
- Estimated total: ~200,000–350,000 tokens
- TPD limit: 500,000 tokens — covered within one day window

## Latency / runtime

- Average API response time per claim: 2–5 seconds
- Inter-call delay: 2 seconds minimum; auto-waits on TPD limit using parsed retry time
- **Estimated total runtime: 8–18 hours** (depends on TPD rolling-window headroom)

## Rate limit and retry strategy

- On 429 / TPD-exceeded: parses `"Please try again in Xm Ys"` from the error message and waits exactly that duration plus 10s buffer
- Retries up to 5 times per claim — handles cases where multiple waits are needed
- On persistent failure: claim marked `not_enough_information` with `manual_review_required` flag
- No parallel calls — sequential processing avoids burst violations

## Image preprocessing

All images resized to maximum 384px on the longest side using Pillow (LANCZOS resampling), saved as JPEG quality=85 before base64 encoding. This reduces token usage ~4–6× vs. full-resolution originals and avoids `invalid image data` errors on very large files.

## Accuracy (sample_claims.csv)

Evaluated per-field against gold labels across 7 key fields:
- **claim_status**, **issue_type**, **object_part**, **severity**, **evidence_standard_met**, **valid_image**, **risk_flags**
- Best measured accuracy: **66%** overall (weighted average across fields)
- Prompt tuned iteratively using Antigravity visual audit findings

Key prompt tuning decisions:
1. `glass_shatter` vs `crack`: require full spider-web epicenter pattern for glass_shatter; default to crack
2. Drawn annotations on images: ignore circles/arrows, assess physical damage only
3. Text instructions in image (sticky notes): → `claim_status=contradicted` + `text_instruction_present` flag
4. Stock photo watermarks (Alamy, Getty): → `valid_image=false` + `non_original_image` flag
5. `contradicted` requires positively seeing part undamaged — not just unclear angle

## Two-pass architecture

1. **Pass 1** (`code/main.py`): vision LLM generates predictions for all claims
2. **Pass 2** (`code/evaluation/auditor.py`): independent Groq vision audit — PASS/WARN/FAIL per row with auto-correction of clear errors

## What was not optimized

- **Parallel async calls** would reduce wall-clock time significantly but risks burst RPM violations on free tier
- **Prompt caching** not available on Groq free tier
- **Smaller model** (llama-3.2-11b-vision) would use fewer tokens but has lower accuracy
