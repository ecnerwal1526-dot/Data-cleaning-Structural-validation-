# Data-cleaning-Structural-validation-
"""
Data Cleaning & Structural Validation Pipeline
================================================
Demonstrates a full cleaning process on a synthetic customer-orders dataset.
Covers:
  1. Generating raw, messy data
  2. Detecting and removing duplicate entries
  3. Handling missing values (per-column strategy)
  4. Standardising inconsistent formats (dates, phone numbers, categories)
  5. Structural / schema validation with per-rule reporting
  6. Exporting the cleaned dataset with an audit log
"""

import pandas as pd
import numpy as np
import re
import hashlib
from datetime import datetime

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE A SYNTHETIC MESSY DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generate_raw_data() -> pd.DataFrame:
    """Create a realistic messy customer-orders dataset."""
    np.random.seed(42)

    raw = pd.DataFrame({
        "order_id": [
            "ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005",
            "ORD-006", "ORD-002",          # ← exact duplicate of row 1
            "ORD-007", "ORD-008", "ORD-009",
            "ORD-010", "ORD-010",          # ← duplicate with minor field diff
            "ORD-011", "ORD-012",
        ],
        "customer_name": [
            "Alice Johnson", "Bob Smith", "Carol White", "david lee", "EVA BROWN",
            "Frank Miller", "Bob Smith",
            "Grace Kim", "HENRY JONES", "Irene Clark",
            "Jack Wilson", "Jack Wilson",
            None,                          # ← missing name
            "Laura Davis",
        ],
        "email": [
            "alice@example.com", "bob@example.com", "carol@example.com",
            "david.lee@example.com", "eva@example.com",
            "frank@example.com", "bob@example.com",
            "grace@example.com", "henry@example.com", "irene@example.com",
            "jack@example.com", "jack@example.com",
            "missing_user@example.com",
            None,                          # ← missing email
        ],
        "phone": [
            "9876543210", "(987) 654-3211", "987-654-3212", "+91 9876543213", "9876543214",
            "9876543215", "(987) 654-3211",
            "9876543216", "987.654.3217", "9876543218",
            "9876543219", "9876543219",
            "987654321",                   # ← too short
            "9876543221",
        ],
        "order_date": [
            "2024-01-15", "15/01/2024", "January 15, 2024", "2024-01-15", "01-15-2024",
            "2024-01-15", "15/01/2024",
            "2024-01-16", "16-01-2024", "2024-01-16",
            "2024-01-17", "2024-01-17",
            "2024-01-17",
            "not-a-date",                  # ← unparseable
        ],
        "amount": [
            250.00, 180.50, None, 320.75, 95.00,   # ← missing amount
            410.00, 180.50,
            560.00, 230.00, None,                   # ← missing amount
            145.00, 145.00,
            None,                                   # ← missing amount
            870.00,
        ],
        "category": [
            "Electronics", "electronics", "ELECTRONICS", "Clothing", "clothing",
            "Furniture", "electronics",
            "Electronics", "CLOTHING", "Furniture",
            "furniture", "furniture",
            "Unknown",                     # ← sentinel value
            "Electronics",
        ],
        "status": [
            "Delivered", "Shipped", "Pending", "delivered", "SHIPPED",
            "Pending", "Shipped",
            "Delivered", "pending", "Cancelled",
            "Delivered", "Delivered",
            "Pending",
            "shipped",
        ],
    })

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# 2. DUPLICATE DETECTION & REMOVAL
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    """
    Two-pass deduplication:
      Pass 1 – exact duplicates across all columns.
      Pass 2 – same order_id with differing field values (keep first occurrence).
    Logs every removal to the audit trail.
    """
    before = len(df)

    # Pass 1: exact row duplicates
    exact_dups = df[df.duplicated(keep=False)]
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    exact_removed = before - len(df)

    # Pass 2: same order_id, different fields (near-duplicate / data-entry error)
    id_dups = df[df.duplicated(subset=["order_id"], keep=False)]
    if not id_dups.empty:
        audit.append({
            "step": "deduplication",
            "detail": f"Near-duplicate order_ids found: {id_dups['order_id'].unique().tolist()}. Keeping first occurrence.",
        })
    df = df.drop_duplicates(subset=["order_id"], keep="first").reset_index(drop=True)
    near_removed = before - exact_removed - len(df)

    audit.append({
        "step": "deduplication",
        "detail": (
            f"Removed {exact_removed} exact duplicate row(s) and "
            f"{near_removed} near-duplicate order_id row(s). "
            f"{len(df)} rows remain."
        ),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MISSING VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    """
    Per-column strategy:
      • amount    → impute with column median (numeric, MAR assumption)
      • email     → fill with placeholder; flag as imputed
      • customer_name → fill with "Unknown Customer"; flag
      • order_date missing/unparseable → will be handled in format step
    All imputed fields get a companion _imputed boolean column.
    """
    # --- amount: median imputation ---
    median_amount = df["amount"].median()
    missing_amount_idx = df["amount"].isna()
    df["amount_imputed"] = False
    df.loc[missing_amount_idx, "amount"] = median_amount
    df.loc[missing_amount_idx, "amount_imputed"] = True
    audit.append({
        "step": "missing_values",
        "detail": (
            f"'amount': {missing_amount_idx.sum()} missing → imputed with median "
            f"({median_amount:.2f}). Rows flagged in 'amount_imputed'."
        ),
    })

    # --- email: placeholder ---
    missing_email_idx = df["email"].isna()
    df["email_imputed"] = False
    df.loc[missing_email_idx, "email"] = "no-email@placeholder.com"
    df.loc[missing_email_idx, "email_imputed"] = True
    audit.append({
        "step": "missing_values",
        "detail": (
            f"'email': {missing_email_idx.sum()} missing → filled with placeholder. "
            "Flagged in 'email_imputed'."
        ),
    })

    # --- customer_name: placeholder ---
    missing_name_idx = df["customer_name"].isna()
    df["name_imputed"] = False
    df.loc[missing_name_idx, "customer_name"] = "Unknown Customer"
    df.loc[missing_name_idx, "name_imputed"] = True
    audit.append({
        "step": "missing_values",
        "detail": (
            f"'customer_name': {missing_name_idx.sum()} missing → filled with "
            "'Unknown Customer'. Flagged in 'name_imputed'."
        ),
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORMAT STANDARDISATION
# ─────────────────────────────────────────────────────────────────────────────

DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
]

def _parse_date(value: str) -> pd.Timestamp | None:
    """Try each known date format; return None if all fail."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue
    return None


def _normalise_phone(phone: str) -> str:
    """Strip all non-digit characters and return a 10-digit string or INVALID."""
    digits = re.sub(r"\D", "", str(phone))
    # Indian numbers may arrive with country code 91
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits if len(digits) == 10 else "INVALID"


def standardise_formats(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    """Normalise dates, phone numbers, categories, statuses, and names."""

    # --- order_date → ISO 8601 ---
    original_dates = df["order_date"].copy()
    df["order_date"] = df["order_date"].apply(_parse_date)
    bad_dates = df["order_date"].isna()
    if bad_dates.any():
        df.loc[bad_dates, "order_date"] = pd.NaT
    audit.append({
        "step": "format_standardisation",
        "detail": (
            f"'order_date': parsed {(~bad_dates).sum()} dates to ISO 8601. "
            f"{bad_dates.sum()} unparseable value(s) set to NaT."
        ),
    })

    # --- phone: digits only, 10-digit validation ---
    df["phone"] = df["phone"].apply(_normalise_phone)
    invalid_phones = (df["phone"] == "INVALID").sum()
    audit.append({
        "step": "format_standardisation",
        "detail": (
            f"'phone': normalised to digits only. "
            f"{invalid_phones} invalid number(s) marked 'INVALID'."
        ),
    })

    # --- customer_name: title case ---
    df["customer_name"] = df["customer_name"].str.title()

    # --- category & status: title case, map sentinel values ---
    df["category"] = df["category"].str.title()
    df["status"] = df["status"].str.title()

    sentinel_mask = df["category"] == "Unknown"
    df.loc[sentinel_mask, "category"] = "Uncategorised"
    audit.append({
        "step": "format_standardisation",
        "detail": (
            f"'category' & 'status': normalised to title case. "
            f"{sentinel_mask.sum()} 'Unknown' category value(s) → 'Uncategorised'."
        ),
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. STRUCTURAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

VALID_CATEGORIES = {"Electronics", "Clothing", "Furniture", "Uncategorised"}
VALID_STATUSES   = {"Delivered", "Shipped", "Pending", "Cancelled"}

def validate_schema(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    """
    Rule-based structural validation. Each rule appends a PASS/FAIL entry.
    Rows that violate hard rules are flagged with a 'validation_flag' column
    rather than silently dropped, preserving data integrity for human review.
    """
    df["validation_flag"] = ""

    def _flag(mask: pd.Series, message: str):
        df.loc[mask, "validation_flag"] += message + "; "

    # Rule 1: order_id must be unique and non-null
    dup_ids = df["order_id"].duplicated()
    null_ids = df["order_id"].isna()
    r1_fail = dup_ids | null_ids
    audit.append({
        "step": "validation",
        "rule": "order_id uniqueness",
        "result": "PASS" if not r1_fail.any() else f"FAIL ({r1_fail.sum()} row(s))",
    })
    if r1_fail.any():
        _flag(r1_fail, "invalid order_id")

    # Rule 2: amount must be positive
    r2_fail = df["amount"] <= 0
    audit.append({
        "step": "validation",
        "rule": "amount > 0",
        "result": "PASS" if not r2_fail.any() else f"FAIL ({r2_fail.sum()} row(s))",
    })
    if r2_fail.any():
        _flag(r2_fail, "non-positive amount")

    # Rule 3: order_date must not be null
    r3_fail = df["order_date"].isna()
    audit.append({
        "step": "validation",
        "rule": "order_date not null",
        "result": "PASS" if not r3_fail.any() else f"FAIL ({r3_fail.sum()} row(s))",
    })
    if r3_fail.any():
        _flag(r3_fail, "unparseable order_date")

    # Rule 4: phone must be 10 digits (not INVALID)
    r4_fail = df["phone"] == "INVALID"
    audit.append({
        "step": "validation",
        "rule": "phone is 10 digits",
        "result": "PASS" if not r4_fail.any() else f"FAIL ({r4_fail.sum()} row(s))",
    })
    if r4_fail.any():
        _flag(r4_fail, "invalid phone")

    # Rule 5: category is one of the allowed values
    r5_fail = ~df["category"].isin(VALID_CATEGORIES)
    audit.append({
        "step": "validation",
        "rule": "category in allowed set",
        "result": "PASS" if not r5_fail.any() else f"FAIL ({r5_fail.sum()} row(s))",
    })
    if r5_fail.any():
        _flag(r5_fail, "unexpected category")

    # Rule 6: status is one of the allowed values
    r6_fail = ~df["status"].isin(VALID_STATUSES)
    audit.append({
        "step": "validation",
        "rule": "status in allowed set",
        "result": "PASS" if not r6_fail.any() else f"FAIL ({r6_fail.sum()} row(s))",
    })
    if r6_fail.any():
        _flag(r6_fail, "unexpected status")

    # Rule 7: email format (basic regex)
    email_pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-z]{2,}$"
    r7_fail = ~df["email"].str.match(email_pattern, na=False)
    audit.append({
        "step": "validation",
        "rule": "email format",
        "result": "PASS" if not r7_fail.any() else f"FAIL ({r7_fail.sum()} row(s))",
    })
    if r7_fail.any():
        _flag(r7_fail, "malformed email")

    # Trim trailing semicolon from flags
    df["validation_flag"] = df["validation_flag"].str.rstrip("; ")

    total_flagged = (df["validation_flag"] != "").sum()
    audit.append({
        "step": "validation",
        "rule": "summary",
        "result": (
            f"{total_flagged} row(s) flagged for review out of {len(df)} total. "
            f"{len(df) - total_flagged} row(s) passed all rules."
        ),
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE ORCHESTRATION & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    audit: list[dict] = []

    print("=" * 70)
    print("DATA CLEANING & STRUCTURAL VALIDATION PIPELINE")
    print("=" * 70)

    # Step 1 – Raw data
    raw = generate_raw_data()
    print(f"\n[1] Raw dataset loaded: {len(raw)} rows × {len(raw.columns)} columns")
    print(raw.to_string(index=False))

    # Step 2 – Deduplication
    df = remove_duplicates(raw.copy(), audit)
    print(f"\n[2] After deduplication: {len(df)} rows")

    # Step 3 – Missing values
    df = handle_missing(df, audit)
    print(f"\n[3] Missing values handled.")

    # Step 4 – Format standardisation
    df = standardise_formats(df, audit)
    print(f"\n[4] Formats standardised.")

    # Step 5 – Structural validation
    df = validate_schema(df, audit)
    print(f"\n[5] Structural validation complete.")

    # ── Final cleaned dataset ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLEANED DATASET")
    print("=" * 70)
    print(df.to_string(index=False))

    # ── Audit log ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AUDIT LOG")
    print("=" * 70)
    audit_df = pd.DataFrame(audit)
    print(audit_df.to_string(index=False))

    # ── Validation summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION RULE RESULTS")
    print("=" * 70)
    val_df = pd.DataFrame([e for e in audit if e.get("step") == "validation" and "rule" in e])
    if not val_df.empty:
        print(val_df[["rule", "result"]].to_string(index=False))

    # ── Flagged rows for human review ──────────────────────────────────────
    flagged = df[df["validation_flag"] != ""]
    if not flagged.empty:
        print("\n" + "=" * 70)
        print("ROWS FLAGGED FOR REVIEW")
        print("=" * 70)
        print(flagged[["order_id", "customer_name", "validation_flag"]].to_string(index=False))

    # ── Export ─────────────────────────────────────────────────────────────
    df.to_csv("/mnt/user-data/outputs/cleaned_orders.csv", index=False)
    audit_df.to_csv("/mnt/user-data/outputs/audit_log.csv", index=False)

    print("\n✓ Exported: cleaned_orders.csv")
    print("✓ Exported: audit_log.csv")
    print("=" * 70)

    return df, audit_df


if __name__ == "__main__":
    run_pipeline()
