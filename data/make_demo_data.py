#!/usr/bin/env python3
"""
data/make_demo_data.py
Generates creator-platform issue dataset for 3-tier priority classification (High, Medium, Low).
Context-dependent labeling based on link status, contacts shared, and creator features.
"""

import os
import sys
import random
import pandas as pd

# Add project root to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.constants import APP_AREAS, PRIORITIES, LINK_STATUSES, DEVICES

random.seed(42)

# Combinable text components (shared across all intents and priorities)
OPENERS = [
    "",
    "hey, ",
    "pls help, ",
    "facing issue: ",
    "hello team, ",
    "urgent, ",
    "can someone check? ",
    "hi support, ",
    "quick question, ",
    "so basically ",
    "need help with this: ",
    "facing a problem, ",
    "check this pls, ",
    "my app is having an issue: ",
    "support team, ",
    "look into this: ",
]

DETAILS = [
    "",
    " tried refreshing and clearing cache.",
    " happening since this morning.",
    " checked with multiple people and same error.",
    " tried restarting the app already.",
    " please resolve quickly.",
    " was working properly yesterday.",
    " happening on my mobile.",
    " my contacts informed me about this.",
    " kindly check what is wrong.",
    " waited for 20 mins still same.",
    " really stuck because of this.",
    " please look into it urgently.",
    " facing this since today morning.",
    " hope you can fix it soon.",
]

TYPOS = [
    ("", ""),
    ("please", "pls"),
    ("friends", "frnds"),
    ("cannot", "cant"),
    ("because", "bcoz"),
    ("problem", "prob"),
    ("thanks", "thx"),
    ("message", "msg"),
    ("working", "workin"),
]

# Intent phrases
LINK_OPENING_PHRASES = [
    "my friends say my link is not opening",
    "shared product link gives error page",
    "contacts telling me link is showing blank screen",
    "my shared link gives 404 not found",
    "people clicking my product link cannot open it",
    "link is loading forever and never opens for my contacts",
    "customers say link is broken",
    "link says page cannot be found",
    "product link shows error when contacts click",
    "frnds say link is not working",
    "sent link to contacts but it shows site cannot be reached",
    "link is completely inaccessible to anyone i send it to",
    "when friends open the product link it shows an error",
    "customers complaining that my link is dead",
    "link just shows white blank screen when opened",
    "buyer tapped my link and it threw connection refused",
    "catalog link is failing to load for anyone on whatsapp",
    "shared url gives server not responding to my friends",
]

LOGIN_LOCKED_PHRASES = [
    "cannot log in, it says account is locked",
    "someone else logged into my account",
    "my account got hacked and i am locked out",
    "unable to sign in, account locked after password try",
    "suspicious activity detected and my account got blocked",
    "all my products and sales data got lost or deleted",
    "lost access to my creator account, says unauthorized",
    "account locked, showing security block",
    "some unknown person changed my profile and logged in",
    "my saved products disappeared completely from my catalog",
    "cannot access my account, says credentials suspended",
    "got logged out and now it says account compromised",
]

PASSWORD_HOWTO_PHRASES = [
    "how do i change my account password",
    "forgot where the reset password option is in settings",
    "how can i update my login email address",
    "need steps to set up two step verification for login",
    "where can i see my login history and active devices",
    "how to switch to another account without losing data",
    "want to update my phone number in account settings",
    "how do i reset my forgotten password pin",
]

SITE_BUILDER_PHRASES = [
    "color i picked is not applying",
    "site builder font style is not changing",
    "theme colors not saving on my store page",
    "banner layout alignment not working in builder",
    "custom background color does not appear on site",
    "site styling reverts back to default after save",
    "button color changed in builder but not showing on page",
    "font style i chose in builder is not reflecting",
    "store template looks messed up after applying modern theme",
    "header title text styling does not save",
    "accent color selected in builder is ignored on mobile preview",
    "custom css styles not applying to product showcase",
]

IMAGE_UPLOAD_PHRASES = [
    "image upload failing when adding new product",
    "cannot upload photos of my items",
    "product photo upload gets stuck at 99%",
    "getting upload error every time i add product pictures",
    "pictures keep failing to upload from my phone gallery",
    "product gallery image upload error keeps showing",
    "uploading second product image fails with unknown error",
    "cannot add new item images in product catalog",
    "product photo upload button freezes and does nothing",
    "uploaded product image becomes grey box after save",
]

BARGAIN_PHRASES = [
    "bargain button missing on product page",
    "bargain feature is broken when customers click",
    "make an offer button does not respond",
    "bargain option is not showing up for my contacts",
    "negotiation bargain button gives error when tapped",
    "customers cannot submit bargain offer on my items",
    "bargain price counter offer button is disabled",
    "bargain discount slider not responding to touches",
    "chat offer button for bargain is missing from my shop",
]

HOWTO_COSMETIC_PHRASES = [
    "how do i add my second product",
    "where can i see the view count on my products",
    "can you add an option to reorder catalog items",
    "would love a dark mode theme for the creator app",
    "how to share store link on whatsapp directly",
    "text spacing on store footer looks slightly tight",
    "how do i archive an unsold product from store",
    "feature request: add customer review section to product page",
    "slight spelling mistake in help center guide",
    "want to know how payment settlement works with bank",
    "how to download qr code for my shop",
    "is there any way to duplicate an existing product listing",
    "can we have bulk price update feature in future update",
    "where to find my store referral link",
]


def apply_typo(text):
    for original, typo in TYPOS:
        if original and original in text and random.random() < 0.35:
            text = text.replace(original, typo, 1)
            break
    return text


def generate_single_sample(intent_idx):
    opener = random.choice(OPENERS)
    detail = random.choice(DETAILS)

    device = random.choice(DEVICES)
    issues_30d = random.choices([0, 1, 2, 3, 4, 5, 7, 10], weights=[0.4, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02])[0]

    # Intent 0: Link not opening
    if intent_idx == 0:
        app_area = "Product Link"
        phrase = random.choice(LINK_OPENING_PHRASES)
        link_status = random.choices(["Live", "Draft", "None"], weights=[0.65, 0.20, 0.15])[0]

        if link_status == "None":
            contacts_shared = 0
            hours_since_link = 0.0
        elif link_status == "Draft":
            contacts_shared = random.choice([0, 0, 0, 1])
            hours_since_link = round(random.uniform(1.0, 72.0), 1)
        else: # Live
            contacts_shared = random.choices(
                [random.randint(10, 80), random.randint(1, 9), 0],
                weights=[0.60, 0.30, 0.10]
            )[0]
            hours_since_link = round(random.uniform(1.0, 168.0), 1)

        # Context-dependent labeling rules
        if link_status == "Live" and contacts_shared >= 10:
            priority = "High"
        elif link_status == "Live" and 1 <= contacts_shared <= 9:
            priority = "High" if random.random() < 0.70 else "Medium"
        else: # Draft or None or Live with 0 contacts
            priority = "Low"

    # Intent 1: Login locked / hacked
    elif intent_idx == 1:
        app_area = "Login & Account"
        phrase = random.choice(LOGIN_LOCKED_PHRASES)
        link_status = random.choice(LINK_STATUSES)
        contacts_shared = random.randint(0, 30) if link_status != "None" else 0
        hours_since_link = round(random.uniform(0.0, 100.0), 1) if link_status != "None" else 0.0
        priority = "High"

    # Intent 2: Password how-to
    elif intent_idx == 2:
        app_area = "Login & Account"
        phrase = random.choice(PASSWORD_HOWTO_PHRASES)
        link_status = random.choice(LINK_STATUSES)
        contacts_shared = random.randint(0, 15) if link_status != "None" else 0
        hours_since_link = round(random.uniform(0.0, 50.0), 1) if link_status != "None" else 0.0
        priority = "Low"

    # Intent 3: Site builder styles not applying
    elif intent_idx == 3:
        app_area = "Site Builder (styles)"
        phrase = random.choice(SITE_BUILDER_PHRASES)
        link_status = random.choices(["Live", "Draft", "None"], weights=[0.55, 0.35, 0.10])[0]
        contacts_shared = random.randint(0, 25) if link_status == "Live" else 0
        hours_since_link = round(random.uniform(1.0, 96.0), 1) if link_status != "None" else 0.0

        if link_status == "Live":
            priority = "Medium"
        else:
            priority = "Low"

    # Intent 4: Image upload failing
    elif intent_idx == 4:
        app_area = "Image / Product Upload"
        phrase = random.choice(IMAGE_UPLOAD_PHRASES)
        link_status = random.choice(LINK_STATUSES)
        contacts_shared = random.randint(0, 20) if link_status != "None" else 0
        hours_since_link = round(random.uniform(0.0, 48.0), 1) if link_status != "None" else 0.0
        priority = "Medium"

    # Intent 5: Bargain button missing/broken
    elif intent_idx == 5:
        app_area = "Bargain Feature"
        phrase = random.choice(BARGAIN_PHRASES)
        link_status = random.choices(["Live", "Draft", "None"], weights=[0.70, 0.20, 0.10])[0]
        if link_status == "Live":
            contacts_shared = random.choices(
                [random.randint(30, 75), random.randint(1, 29)],
                weights=[0.35, 0.65]
            )[0]
            hours_since_link = round(random.uniform(1.0, 120.0), 1)
        else:
            contacts_shared = 0
            hours_since_link = round(random.uniform(1.0, 48.0), 1) if link_status != "None" else 0.0

        if link_status == "Live" and contacts_shared >= 30:
            priority = "High"
        else:
            priority = "Medium"

    # Intent 6: How-to, cosmetic, feature request
    else:
        app_area = random.choice(["Other", "Site Builder (styles)", "Product Link", "Image / Product Upload"])
        phrase = random.choice(HOWTO_COSMETIC_PHRASES)
        link_status = random.choice(LINK_STATUSES)
        contacts_shared = random.randint(0, 25) if link_status != "None" else 0
        hours_since_link = round(random.uniform(0.0, 80.0), 1) if link_status != "None" else 0.0
        priority = "Low"

    # Assemble text
    description = f"{opener}{phrase}{detail}".strip()
    description = apply_typo(description)

    # Flip 5% of labels to neighbouring level to simulate human disagreement
    if random.random() < 0.05:
        if priority == "High":
            priority = "Medium"
        elif priority == "Low":
            priority = "Medium"
        else:  # Medium
            priority = random.choice(["High", "Low"])

    return {
        "description": description,
        "app_area": app_area,
        "link_status": link_status,
        "contacts_shared": contacts_shared,
        "hours_since_link_created": hours_since_link,
        "device": device,
        "issues_last_30d": issues_30d,
        "priority": priority
    }


def generate_dataset(target_rows=1800):
    rows = []
    # Balance across intents
    # Intent weights:
    # 0 (link): 0.28, 1 (login high): 0.14, 2 (pwd low): 0.08,
    # 3 (builder): 0.16, 4 (image): 0.14, 5 (bargain): 0.10, 6 (howto/cosmetic): 0.10
    intents = [0, 1, 2, 3, 4, 5, 6]
    intent_weights = [0.28, 0.14, 0.08, 0.16, 0.14, 0.10, 0.10]

    for _ in range(target_rows):
        chosen_intent = random.choices(intents, weights=intent_weights, k=1)[0]
        sample = generate_single_sample(chosen_intent)
        rows.append(sample)

    df = pd.DataFrame(rows)
    return df


def main():
    output_path = os.path.join(BASE_DIR, "data", "app_issues_demo.csv")
    df = generate_dataset(target_rows=1800)

    total_rows = len(df)
    unique_descriptions = df["description"].nunique()
    unique_ratio = unique_descriptions / total_rows

    # Assertions for compliance
    assert total_rows >= 1500, f"Expected at least 1500 rows, got {total_rows}"
    assert unique_ratio >= 0.90, f"Expected unique ratio >= 0.90, got {unique_ratio:.4f}"
    assert "Critical" not in df["priority"].values, "Found 'Critical' in priority labels!"
    for p in df["priority"].unique():
        assert p in PRIORITIES, f"Unexpected priority {p}"

    df.to_csv(output_path, index=False)

    print(f"Dataset generated successfully at: {output_path}")
    print(f"Total Rows: {total_rows}")
    print(f"Unique Descriptions: {unique_descriptions} ({unique_ratio:.2%})")
    print("\nClass Counts:")
    print(df["priority"].value_counts())
    print("\nSample Rows:")
    print(df[["description", "app_area", "link_status", "contacts_shared", "priority"]].head(5))


if __name__ == "__main__":
    main()
