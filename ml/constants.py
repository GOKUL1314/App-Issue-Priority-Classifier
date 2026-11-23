"""
ml/constants.py
Pure Python constants for ML models, priority classification, and domain features.
No Django imports.
"""

APP_AREAS = [
    "Product Link",
    "Site Builder (styles)",
    "Login & Account",
    "Image / Product Upload",
    "Bargain Feature",
    "Other"
]

PRIORITIES = [
    "High",
    "Medium",
    "Low"
]

LINK_STATUSES = [
    "Live",
    "Draft",
    "None"
]

DEVICES = [
    "Android",
    "iOS",
    "Desktop"
]

TEXT_FEATURE = "description"
CATEGORICAL_FEATURES = ["app_area", "link_status", "device"]
NUMERIC_FEATURES = ["contacts_shared", "hours_since_link_created", "issues_last_30d"]
ALL_FEATURE_COLUMNS = [TEXT_FEATURE] + CATEGORICAL_FEATURES + NUMERIC_FEATURES
LABEL_COLUMN = "priority"
