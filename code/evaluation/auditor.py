"""
Antigravity-style second-pass auditor.
Reviews each prediction row against its submitted images using Groq vision.
Produces a PASS / WARN / FAIL report and optionally corrects rows.

Usage:
    python code/evaluation/auditor.py --predictions evaluation/eval_output.csv
    python code/evaluation/auditor.py --predictions dataset/output.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    REPO_ROOT, MODEL,
    OBJECT_PARTS, RISK_FLAG_VALUES,
    CLAIM_STATUS_VALUES, ISSUE_TYPE_VALUES, SEVERITY_VALUES,
    encode_image, resize_and_encode,
)

AUDITOR_SYSTEM = """You are an independent damage claim auditor. You receive a prediction row
produced by an automated system and the original images. Verify whether the prediction is correct.

=== VALID_IMAGE REMINDER ===
valid_image = TRUE  = genuine original photo (not AI-generated or tampered).
valid_image = FALSE = AI-generated, stock/watermarked photo, or digitally manipulated.
A blurry or wrong-angle real photo is still valid_image=true.

=== AUDIT CHECKS ===
1. claim_status: Does supported/contradicted/not_enough_information match what you see?
2. evidence_standard_met: Consistent with valid_image and claim_status?
3. issue_type: Matches visible damage?
4. object_part: Matches visible part and claim?
5. risk_flags: Any missed flags?
6. severity: Proportionate to visible damage?
7. supporting_image_ids: Do cited IDs actually show the damage?
8. Hallucination: Does justification describe things not visible?

=== OUTPUT FORMAT ===
Return ONLY valid JSON:
{
  "verdict": "PASS" | "WARN" | "FAIL",
  "issues": ["field: explanation"],
  "corrected": { "field": "corrected_value" },
  "notes": "brief explanation grounded in what you see"
}
Omit "corrected" if verdict is PASS.
FAIL = claim_status clearly wrong, evidence_standard_met=true with valid_image=false, wrong part type, empty justification."""


def audit_row(client: Groq, pred_row: dict, retries: int = 3) -> dict:
    import base64
    image_paths_raw = [p.strip() for p in pred_row["image_paths"].split(";")]
    image_ids = [Path(p).stem for p in image_paths_raw]

    content = []
    for img_path_str, img_id in zip(image_paths_raw, image_ids):
        full_path = REPO_ROOT / "dataset" / img_path_str
        if not full_path.exists():
            full_path = REPO_ROOT / img_path_str
        content.append({"type": "text", "text": f"Image ID: {img_id}"})
        if not full_path.exists():
            content.append({"type": "text", "text": f"[Image {img_id}: not found]"})
            continue
        b64, mime_type = resize_and_encode(full_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})

    content.append({"type": "text", "text": (
        f"\n## Prediction to audit\n"
        f"User ID: {pred_row['user_id']} | Claim object: {pred_row['claim_object']}\n"
        f"Conversation: {pred_row['user_claim']}\n\n"
        f"claim_status:               {pred_row.get('claim_status')}\n"
        f"claim_status_justification: {pred_row.get('claim_status_justification')}\n"
        f"evidence_standard_met:      {pred_row.get('evidence_standard_met')}\n"
        f"valid_image:                {pred_row.get('valid_image')}\n"
        f"issue_type:                 {pred_row.get('issue_type')}\n"
        f"object_part:                {pred_row.get('object_part')}\n"
        f"risk_flags:                 {pred_row.get('risk_flags')}\n"
        f"severity:                   {pred_row.get('severity')}\n"
        f"supporting_image_ids:       {pred_row.get('supporting_image_ids')}\n\n"
        f"Image IDs: {', '.join(image_ids)}\nReturn JSON only."
    )})

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": AUDITOR_SYSTEM},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            err = str(e)
            wait = 60 * (attempt + 1) if ("429" in err or "rate" in err.lower()) else 5
            print(f"    Auditor error attempt {attempt+1}: {e}", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(wait)

    return {"verdict": "WARN", "issues": ["auditor error"], "notes": "Could not audit."}


def apply_corrections(pred_row: dict, audit: dict) -> dict:
    corrected = audit.get("corrected", {})
    if not corrected:
        return pred_row
    row = dict(pred_row)
    claim_object = row.get("claim_object", "")
    for field, value in corrected.items():
        if field not in row:
            continue
        if field == "claim_status" and value not in CLAIM_STATUS_VALUES:
            continue
        if field == "issue_type" and value not in ISSUE_TYPE_VALUES:
            continue
        if field == "object_part" and value not in OBJECT_PARTS.get(claim_object, {"unknown"}):
            continue
        if field == "severity" and value not in SEVERITY_VALUES:
            continue
        if field in ("evidence_standard_met", "valid_image"):
            value = str(value).lower()
        if field == "risk_flags" and isinstance(value, list):
            value = ";".join(v for v in value if v in RISK_FLAG_VALUES) or "none"
        if field == "supporting_image_ids" and isinstance(value, list):
            value = ";".join(value)
        row[field] = str(value)
    return row


def run_audit(predictions_path: Path, apply_fixes: bool = True, delay: float = 2.0):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: GROQ_API_KEY not set. Get a free key at https://console.groq.com")

    client = Groq(api_key=api_key)
    with open(predictions_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\nAuditing {len(rows)} predictions from {predictions_path}...")
    audit_results, corrected_rows = [], []
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    for i, row in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] {row['user_id']} -- ", end="", flush=True)
        audit = audit_row(client, row)
        verdict = audit.get("verdict", "WARN")
        counts[verdict] = counts.get(verdict, 0) + 1
        print(verdict)
        for issue in audit.get("issues", []):
            print(f"    - {issue}")
        audit_results.append({
            "user_id": row["user_id"],
            "verdict": verdict,
            "issues": "; ".join(audit.get("issues", [])),
            "notes": audit.get("notes", ""),
        })
        corrected_rows.append(apply_corrections(row, audit) if apply_fixes else row)
        if i < len(rows):
            time.sleep(delay)

    print(f"\nAudit: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
    return audit_results, corrected_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--audit-report", default="dataset/audit_report.csv")
    parser.add_argument("--no-fix", action="store_true")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    predictions_path = REPO_ROOT / args.predictions
    output_path = REPO_ROOT / (args.output or args.predictions)
    report_path = REPO_ROOT / args.audit_report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    audit_results, corrected_rows = run_audit(predictions_path, not args.no_fix, args.delay)

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "verdict", "issues", "notes"])
        writer.writeheader()
        writer.writerows(audit_results)
    print(f"Audit report -> {report_path}")

    if not args.no_fix and corrected_rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(corrected_rows[0].keys()))
            writer.writeheader()
            writer.writerows(corrected_rows)
        print(f"Corrected predictions -> {output_path}")


if __name__ == "__main__":
    main()
