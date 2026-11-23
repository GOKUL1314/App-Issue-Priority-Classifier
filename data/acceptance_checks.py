#!/usr/bin/env python3
"""
data/acceptance_checks.py
Executes acceptance check (c) Context tests and (d) Unseen 8 sentences tests.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.predict import score_issue


def run_check_c():
    print("=" * 70)
    print("ACCEPTANCE CHECK (c): Context sensitivity tests")
    print("=" * 70)

    # Pair 1: "my friends say my link is not opening"
    text1 = "my friends say my link is not opening"
    res1_live = score_issue(
        description=text1,
        app_area="Product Link",
        link_status="Live",
        contacts_shared=40,
        hours_since_link_created=12.0,
        device="Android",
        issues_last_30d=1
    )
    res1_draft = score_issue(
        description=text1,
        app_area="Product Link",
        link_status="Draft",
        contacts_shared=0,
        hours_since_link_created=0.0,
        device="Android",
        issues_last_30d=1
    )

    print(f"\nSentence 1: '{text1}'")
    print(f"  Live / 40 contacts  -> ML: {res1_live['ml_prediction']:6} | Final: {res1_live['final_priority']:6} | Confidence: {res1_live['confidence']}% | Source: {res1_live['priority_source']} | Rule: {res1_live['rule_name']}")
    print(f"  Draft / 0 contacts  -> ML: {res1_draft['ml_prediction']:6} | Final: {res1_draft['final_priority']:6} | Confidence: {res1_draft['confidence']}% | Source: {res1_draft['priority_source']} | Rule: {res1_draft['rule_name']}")
    assert res1_live['final_priority'] != res1_draft['final_priority'], "Context did not change result for Sentence 1!"

    # Pair 2: "color i picked is not applying"
    text2 = "color i picked is not applying"
    res2_live = score_issue(
        description=text2,
        app_area="Site Builder (styles)",
        link_status="Live",
        contacts_shared=20,
        hours_since_link_created=24.0,
        device="Desktop",
        issues_last_30d=0
    )
    res2_draft = score_issue(
        description=text2,
        app_area="Site Builder (styles)",
        link_status="Draft",
        contacts_shared=0,
        hours_since_link_created=2.0,
        device="Desktop",
        issues_last_30d=0
    )

    print(f"\nSentence 2: '{text2}'")
    print(f"  Live / 20 contacts  -> ML: {res2_live['ml_prediction']:6} | Final: {res2_live['final_priority']:6} | Confidence: {res2_live['confidence']}% | Source: {res2_live['priority_source']} | Rule: {res2_live['rule_name']}")
    print(f"  Draft / 0 contacts  -> ML: {res2_draft['ml_prediction']:6} | Final: {res2_draft['final_priority']:6} | Confidence: {res2_draft['confidence']}% | Source: {res2_draft['priority_source']} | Rule: {res2_draft['rule_name']}")
    assert res2_live['final_priority'] != res2_draft['final_priority'], "Context did not change result for Sentence 2!"
    print("\nResult: Context successfully changes priority for both test sentences!\n")


def run_check_d():
    print("=" * 70)
    print("ACCEPTANCE CHECK (d): Test 8 unseen sentences NOT in the generator")
    print("=" * 70)

    unseen_test_cases = [
        {
            "desc": "page keeps loading forever for my cousins",
            "area": "Product Link",
            "link_status": "Live",
            "contacts": 35,
            "hours": 10.0,
            "device": "Android",
            "expected_intent": "Link broken for contacts -> High"
        },
        {
            "desc": "forgot how to add my second product",
            "area": "Other",
            "link_status": "None",
            "contacts": 0,
            "hours": 0.0,
            "device": "Desktop",
            "expected_intent": "How-to question -> Low"
        },
        {
            "desc": "someone else logged into my account",
            "area": "Login & Account",
            "link_status": "Live",
            "contacts": 10,
            "hours": 20.0,
            "device": "Android",
            "expected_intent": "Account hacked / security breach -> High"
        },
        {
            "desc": "button to negotiate bargain price is completely missing",
            "area": "Bargain Feature",
            "link_status": "Live",
            "contacts": 15,
            "hours": 8.0,
            "device": "iOS",
            "expected_intent": "Bargain feature broken with <30 contacts -> Medium"
        },
        {
            "desc": "picture of my ceramic mug fails while uploading",
            "area": "Image / Product Upload",
            "link_status": "Live",
            "contacts": 5,
            "hours": 14.0,
            "device": "Android",
            "expected_intent": "Image upload failing -> Medium"
        },
        {
            "desc": "can you please add an option to duplicate product listings",
            "area": "Other",
            "link_status": "Draft",
            "contacts": 0,
            "hours": 5.0,
            "device": "Desktop",
            "expected_intent": "Feature request -> Low"
        },
        {
            "desc": "all my saved items got deleted from my store",
            "area": "Login & Account",
            "link_status": "Live",
            "contacts": 25,
            "hours": 48.0,
            "device": "Android",
            "expected_intent": "Product / data lost or deleted -> High"
        },
        {
            "desc": "store theme background color doesn't change on live site",
            "area": "Site Builder (styles)",
            "link_status": "Live",
            "contacts": 18,
            "hours": 36.0,
            "device": "Desktop",
            "expected_intent": "Style not applying on Live -> Medium"
        }
    ]

    for i, test in enumerate(unseen_test_cases, 1):
        res = score_issue(
            description=test["desc"],
            app_area=test["area"],
            link_status=test["link_status"],
            contacts_shared=test["contacts"],
            hours_since_link_created=test["hours"],
            device=test["device"],
            issues_last_30d=1
        )
        print(f"\nSentence {i}: \"{test['desc']}\"")
        print(f"  Context: Area='{test['area']}', Link='{test['link_status']}', Contacts={test['contacts']}, Device='{test['device']}'")
        print(f"  ML Model Prediction : {res['ml_prediction']} (Confidence: {res['confidence']}%)")
        print(f"  Final Priority       : {res['final_priority']}")
        print(f"  Priority Source      : {res['priority_source']} (Rule: {res['rule_name']})")
        print(f"  Target Urgency       : {test['expected_intent']}")


if __name__ == "__main__":
    run_check_c()
    run_check_d()
