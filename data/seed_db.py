#!/usr/bin/env python
"""
data/seed_db.py
Seeds initial demo creator accounts and sample app issues
in the creator-platform domain with context fields and 3 priority levels.
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app_issue_priority.settings')
django.setup()

from Creator_Portal.models import CreatorAccount, AppIssue
from ml.predict import score_issue


def seed():
    print("Seeding database with creator accounts and realistic platform issues...")

    # Create demo creator 1
    creator, created = CreatorAccount.objects.get_or_create(
        username="tech_creator",
        defaults={
            "email": "creator@example.com",
            "full_name": "Sarah Connor",
            "channel_or_app_name": "Sarah Crafts"
        }
    )
    if created:
        creator.set_password("creator123")
        creator.save()
        print("Created demo creator: username='tech_creator', password='creator123'")
    else:
        print("Demo creator 'tech_creator' already exists.")

    # Create demo creator 2
    creator2, created2 = CreatorAccount.objects.get_or_create(
        username="game_streamer",
        defaults={
            "email": "streamer@example.com",
            "full_name": "Marcus Vance",
            "channel_or_app_name": "Pixel Merch"
        }
    )
    if created2:
        creator2.set_password("streamer123")
        creator2.save()
        print("Created demo creator: username='game_streamer', password='streamer123'")
    else:
        print("Demo creator 'game_streamer' already exists.")

    sample_issues = [
        {
            "creator": creator,
            "title": "Contacts saying product link gives 404 error",
            "product_name": "Handmade Clay Pottery",
            "app_area": "Product Link",
            "description": "My friends and family say the link is not opening and showing 404 error page. Sent to 45 contacts on WhatsApp.",
            "link_status": "Live",
            "contacts_shared": 45,
            "hours_since_link_created": 18.5,
            "device": "Android",
            "issues_last_30d": 1,
            "status": "In Progress",
            "manager_notes": "Routing issue for custom domain link identified and fixing."
        },
        {
            "creator": creator,
            "title": "Cannot log in, says account is locked",
            "product_name": "",
            "app_area": "Login & Account",
            "description": "Cannot log in, it says account is locked after incorrect attempt. Need urgent access to check my sales.",
            "link_status": "Live",
            "contacts_shared": 12,
            "hours_since_link_created": 30.0,
            "device": "Android",
            "issues_last_30d": 0,
            "status": "In Review",
            "manager_notes": "Verification link sent to registered email."
        },
        {
            "creator": creator2,
            "title": "Custom theme color not applying to product showcase",
            "product_name": "Gaming Desk Mat",
            "app_area": "Site Builder (styles)",
            "description": "The accent color i picked in site builder is not applying on my live store page. Reverts back to white.",
            "link_status": "Live",
            "contacts_shared": 20,
            "hours_since_link_created": 40.0,
            "device": "Desktop",
            "issues_last_30d": 2,
            "status": "Open",
            "manager_notes": ""
        },
        {
            "creator": creator2,
            "title": "Product gallery photo upload stuck at 99%",
            "product_name": "Custom Keycaps Set",
            "app_area": "Image / Product Upload",
            "description": "Image upload failing when adding new product pictures from my phone gallery. Upload spinner freezes at 99%.",
            "link_status": "Live",
            "contacts_shared": 15,
            "hours_since_link_created": 12.0,
            "device": "Android",
            "issues_last_30d": 1,
            "status": "Open",
            "manager_notes": ""
        },
        {
            "creator": creator,
            "title": "Bargain button missing on live product page",
            "product_name": "Embroidered Tote Bag",
            "app_area": "Bargain Feature",
            "description": "Bargain feature is broken when customers click make an offer button. Shared with 35 family members.",
            "link_status": "Live",
            "contacts_shared": 35,
            "hours_since_link_created": 24.0,
            "device": "iOS",
            "issues_last_30d": 0,
            "status": "In Progress",
            "manager_notes": "Feature flag checked for merchant tier."
        },
        {
            "creator": creator2,
            "title": "How do I change my account password",
            "product_name": "",
            "app_area": "Login & Account",
            "description": "Where can I reset my account password? Looking for the password update option in profile settings.",
            "link_status": "Live",
            "contacts_shared": 5,
            "hours_since_link_created": 50.0,
            "device": "Desktop",
            "issues_last_30d": 0,
            "status": "Resolved",
            "manager_notes": "Directed user to Account Settings > Security."
        },
        {
            "creator": creator,
            "title": "Font spacing on store header slightly tight",
            "product_name": "Handmade Clay Pottery",
            "app_area": "Site Builder (styles)",
            "description": "Text spacing between store title and description looks slightly tight on mobile screen. Could letter spacing be improved?",
            "link_status": "Draft",
            "contacts_shared": 0,
            "hours_since_link_created": 4.0,
            "device": "iOS",
            "issues_last_30d": 0,
            "status": "Resolved",
            "manager_notes": "Passed feedback to design team."
        },
        {
            "creator": creator2,
            "title": "How to add multiple product variants in draft",
            "product_name": "RGB LED Strip",
            "app_area": "Other",
            "description": "Quick question on how do i add my second product variant with different color choices in draft mode?",
            "link_status": "Draft",
            "contacts_shared": 0,
            "hours_since_link_created": 2.0,
            "device": "Desktop",
            "issues_last_30d": 0,
            "status": "Open",
            "manager_notes": ""
        }
    ]

    for item in sample_issues:
        score = score_issue(
            description=item["description"],
            app_area=item["app_area"],
            link_status=item["link_status"],
            contacts_shared=item["contacts_shared"],
            hours_since_link_created=item["hours_since_link_created"],
            device=item["device"],
            issues_last_30d=item["issues_last_30d"]
        )

        issue, _ = AppIssue.objects.get_or_create(
            title=item["title"],
            defaults={
                "creator": item["creator"],
                "product_name": item["product_name"],
                "description": item["description"],
                "app_area": item["app_area"],
                "link_status": item["link_status"],
                "contacts_shared": item["contacts_shared"],
                "hours_since_link_created": item["hours_since_link_created"],
                "device": item["device"],
                "issues_last_30d": item["issues_last_30d"],
                "predicted_priority": score["ml_prediction"],
                "final_priority": score["final_priority"],
                "confidence": score["confidence"],
                "priority_source": score["priority_source"],
                "rule_name": score["rule_name"],
                "status": item["status"],
                "manager_notes": item["manager_notes"]
            }
        )
        print(f"Created Issue #{issue.id}: [{issue.final_priority}] {issue.title} (Source: {issue.priority_source})")

    print("\nDatabase seeded successfully!")


if __name__ == '__main__':
    seed()
