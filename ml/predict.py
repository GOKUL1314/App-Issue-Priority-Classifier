#!/usr/bin/env python3
"""
ml/predict.py
Loads the saved best ML pipeline, scores individual issues with context,
and enforces platform safety rules that only RAISE priority.
"""

import os
import re
import sys
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.constants import (
    PRIORITIES,
    TEXT_FEATURE,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES
)

MODEL_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "priority_model.joblib")

# Priority numeric ranking for priority comparison (High=3, Medium=2, Low=1)
PRIORITY_RANK = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

# Word-boundary regexes for platform rules that only RAISE priority
# Rule A: Product link not opening / error / blank / 404 on Live link with contacts shared >= 1
RE_LINK_BROKEN = re.compile(
    r"\b(not\s+open(ing)?|cannot\s+open|can\'?t\s+open|won\'?t\s+open|error|blank|404|broken|dead\s+link|not\s+work(ing)?|unreachable|fails?\s+to\s+load|loading\s+forever)\b",
    re.IGNORECASE
)

# Rule B: Cannot log in, account locked, hacked, someone else in account, product or data deleted or lost
RE_ACCOUNT_SECURITY = re.compile(
    r"\b(cannot\s+log\s*in|can\'?t\s+log\s*in|unable\s+to\s+log\s*in|account\s+locked|locked\s+out|hacked|someone\s+else\s+(logged\s+in|in\s+my\s+account)|((product|data|catalog)\s+.*(deleted|lost|disappeared))|(lost|deleted)\s+.*(product|data))\b",
    re.IGNORECASE
)


class PriorityPredictor:
    _pipeline = None

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None and os.path.exists(MODEL_PATH):
            try:
                cls._pipeline = joblib.load(MODEL_PATH)
            except Exception as e:
                print(f"Warning: Could not load model from {MODEL_PATH}: {e}")
                cls._pipeline = None
        return cls._pipeline

    @classmethod
    def reload_model(cls):
        """Force reloads model pipeline after re-training."""
        if os.path.exists(MODEL_PATH):
            cls._pipeline = joblib.load(MODEL_PATH)
        else:
            cls._pipeline = None
        return cls._pipeline

    @classmethod
    def is_trained(cls):
        return os.path.exists(MODEL_PATH)

    @classmethod
    def evaluate_raising_rules(
        cls,
        description: str,
        app_area: str,
        link_status: str,
        contacts_shared: float
    ):
        """
        Evaluates domain platform rules that can only RAISE priority to High.
        Returns (rule_matched, target_priority, rule_name).
        """
        desc_clean = (description or "").strip()

        # Rule (a): app_area == "Product Link" and text suggests not opening/error/blank/404 and link_status == "Live" and contacts_shared >= 1
        if (
            app_area == "Product Link"
            and link_status == "Live"
            and contacts_shared >= 1
            and RE_LINK_BROKEN.search(desc_clean)
        ):
            return True, "High", "RULE_LIVE_PRODUCT_LINK_BROKEN"

        # Rule (b): text about cannot log in, account locked, hacked, someone else in account, product/data deleted or lost
        if RE_ACCOUNT_SECURITY.search(desc_clean):
            return True, "High", "RULE_ACCOUNT_SECURITY_OR_DATA_LOSS"

        return False, None, "None"

    @classmethod
    def score_issue(
        cls,
        description: str,
        app_area: str = "Other",
        link_status: str = "None",
        contacts_shared: float = 0,
        hours_since_link_created: float = 0,
        device: str = "Desktop",
        issues_last_30d: float = 0
    ):
        """
        Scores an app issue and returns structured diagnosis.
        Signature: score_issue(description, app_area, link_status, contacts_shared,
                               hours_since_link_created, device, issues_last_30d)
        """
        description = (description or "").strip()
        app_area = (app_area or "Other").strip()
        link_status = (link_status or "None").strip()
        device = (device or "Desktop").strip()

        try:
            contacts_shared = float(contacts_shared)
        except (ValueError, TypeError):
            contacts_shared = 0.0

        try:
            hours_since_link_created = float(hours_since_link_created)
        except (ValueError, TypeError):
            hours_since_link_created = 0.0

        try:
            issues_last_30d = float(issues_last_30d)
        except (ValueError, TypeError):
            issues_last_30d = 0.0

        pipeline = cls.get_pipeline()

        # Evaluate platform raising rules first
        rule_matched, target_priority, matched_rule_name = cls.evaluate_raising_rules(
            description=description,
            app_area=app_area,
            link_status=link_status,
            contacts_shared=contacts_shared
        )

        if pipeline is not None:
            input_df = pd.DataFrame([{
                TEXT_FEATURE: description,
                "app_area": app_area,
                "link_status": link_status,
                "device": device,
                "contacts_shared": contacts_shared,
                "hours_since_link_created": hours_since_link_created,
                "issues_last_30d": issues_last_30d
            }])

            # Predict priority class
            ml_pred = str(pipeline.predict(input_df)[0])

            # Probabilities / confidence
            probabilities = {}
            confidence = 0.85
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(input_df)[0]
                classes = list(pipeline.classes_)
                probabilities = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}
                confidence = float(np.max(probs))
            else:
                probabilities = {ml_pred: 1.0}

            # Rules may ONLY RAISE a priority, never lower it
            if rule_matched and PRIORITY_RANK.get(target_priority, 0) > PRIORITY_RANK.get(ml_pred, 0):
                final_priority = target_priority
                priority_source = "rule"
                rule_name = matched_rule_name
            else:
                final_priority = ml_pred
                priority_source = "model"
                rule_name = matched_rule_name if rule_matched else "None"

            return {
                "ml_prediction": ml_pred,
                "final_priority": final_priority,
                "confidence": round(confidence * 100, 2),
                "probabilities": probabilities,
                "priority_source": priority_source,
                "rule_name": rule_name
            }
        else:
            # If no model file exists: apply the rules, otherwise Medium, and set priority_source="rules_only"
            # Never present that as an ML result
            if rule_matched and target_priority == "High":
                final_priority = "High"
                rule_name = matched_rule_name
            else:
                final_priority = "Medium"
                rule_name = "None"

            return {
                "ml_prediction": "N/A",
                "final_priority": final_priority,
                "confidence": 0.0,
                "probabilities": {},
                "priority_source": "rules_only",
                "rule_name": rule_name
            }


def score_issue(
    description: str,
    app_area: str = "Other",
    link_status: str = "None",
    contacts_shared: float = 0,
    hours_since_link_created: float = 0,
    device: str = "Desktop",
    issues_last_30d: float = 0
):
    """Module-level score_issue wrapper."""
    return PriorityPredictor.score_issue(
        description=description,
        app_area=app_area,
        link_status=link_status,
        contacts_shared=contacts_shared,
        hours_since_link_created=hours_since_link_created,
        device=device,
        issues_last_30d=issues_last_30d
    )


if __name__ == "__main__":
    print("Testing score_issue with context:")
    test1 = score_issue(
        description="my friends say my link is not opening",
        app_area="Product Link",
        link_status="Live",
        contacts_shared=40
    )
    print("Test 1 (Live/40 contacts):", test1)

    test2 = score_issue(
        description="my friends say my link is not opening",
        app_area="Product Link",
        link_status="Draft",
        contacts_shared=0
    )
    print("Test 2 (Draft/0 contacts):", test2)
