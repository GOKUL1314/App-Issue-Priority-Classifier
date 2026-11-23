#!/usr/bin/env python3
"""
data/check_flow.py
Executes Acceptance Check (f):
- Submits as a creator (verifies only ticket number is shown)
- Finds it in manager queue
- Changes its priority (sets priority_source="manager")
- Retrains model
- Confirms the train page / metrics report 1 correction included.
"""

import os
import sys
import json

os.environ.setdefault('MANAGER_USER', 'admin')
os.environ.setdefault('MANAGER_PASS', 'admin123')

import django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app_issue_priority.settings')
django.setup()

from django.test.utils import setup_test_environment
setup_test_environment()

from django.test import Client
from django.urls import reverse
from django.contrib.messages import get_messages
from Creator_Portal.models import CreatorAccount, AppIssue
from ml.train import METRICS_PATH


def run_check_f():
    print("=" * 70)
    print("ACCEPTANCE CHECK (f): End-to-end flow test")
    print("=" * 70)

    client = Client()

    # Step 1: Log in as creator
    print("\n[Step 1] Logging in as creator 'tech_creator'...")
    login_resp = client.post(reverse('creator_login'), {
        'username': 'tech_creator',
        'password': 'creator123'
    }, follow=True)
    assert login_resp.status_code == 200, "Creator login failed!"
    print("  -> Creator logged in successfully.")

    # Step 2: Submit an issue as creator
    print("\n[Step 2] Submitting issue as creator...")
    submit_resp = client.post(reverse('customer_care'), {
        'app_area': 'Site Builder (styles)',
        'description': 'Banner styling in builder is misaligned for my holiday promo sale',
        'product_name': 'Holiday Mug',
        'title': 'Holiday sale banner styling broken',
        'link_status': 'Live',
        'contacts_shared': '15',
        'hours_since_link_created': '8.0',
        'device': 'Android',
        'issues_last_30d': '1'
    }, follow=True)

    # Verify message shown to creator
    messages = list(get_messages(submit_resp.wsgi_request))
    msg_texts = [str(m) for m in messages]
    print(f"  -> Messages received by creator: {msg_texts}")
    assert any("Complaint received. Ticket #" in m for m in msg_texts), "Expected 'Complaint received. Ticket #' message!"
    assert not any("Priority:" in m or "High" in m or "Medium" in m or "Rule triggered" in m for m in msg_texts), "Priority leaked to creator!"

    # Get the created issue
    created_issue = AppIssue.objects.filter(title='Holiday sale banner styling broken').latest('id')
    print(f"  -> Ticket #{created_issue.id} created. Initial priority: {created_issue.final_priority}, Priority source: {created_issue.priority_source}")

    # Log out creator
    client.post(reverse('creator_logout'))

    # Step 3: Log in as manager and find ticket in queue
    print("\n[Step 3] Logging in as Care Manager and inspecting queue...")
    mgr_login = client.post(reverse('manager_login'), {
        'username': 'admin',
        'password': 'admin123'
    }, follow=True)
    assert mgr_login.status_code == 200, "Manager login failed!"

    queue_resp = client.get(reverse('manager_issues'))
    assert queue_resp.status_code == 200, f"Expected 200 from manager_issues, got {queue_resp.status_code}"
    issues_in_queue = list(queue_resp.context['issues'])
    queue_ids = [i.id for i in issues_in_queue]
    assert created_issue.id in queue_ids, f"Ticket #{created_issue.id} not found in manager queue!"
    print(f"  -> Ticket #{created_issue.id} found in manager prioritized queue at position {queue_ids.index(created_issue.id) + 1} of {len(queue_ids)}.")

    # Step 4: Manager changes priority
    print(f"\n[Step 4] Manager changes Ticket #{created_issue.id} priority from '{created_issue.final_priority}' to 'High'...")
    update_resp = client.post(reverse('update_issue_status', args=[created_issue.id]), {
        'status': 'In Progress',
        'final_priority': 'High',
        'manager_notes': 'Escalated by manager for holiday sale urgency'
    }, follow=True)
    assert update_resp.status_code == 200

    created_issue.refresh_from_db()
    print(f"  -> Updated Issue #{created_issue.id}:")
    print(f"     final_priority    : {created_issue.final_priority}")
    print(f"     predicted_priority: {created_issue.predicted_priority}")
    print(f"     priority_source   : {created_issue.priority_source}")
    print(f"     corrected_at      : {created_issue.corrected_at}")
    assert created_issue.final_priority == 'High'
    assert created_issue.priority_source == 'manager'
    assert created_issue.corrected_at is not None

    # Step 5: Retrain model via manager portal
    print("\n[Step 5] Triggering model retraining via manager portal...")
    train_resp = client.post(reverse('manager_train'), follow=True)
    assert train_resp.status_code == 200

    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    corrections_used = metrics.get('corrections_used', 0)
    print(f"  -> Retraining complete! Corrections used: {corrections_used}")
    print(f"  -> Best model: {metrics['best_algorithm']} (F1: {metrics['best_f1_score']}%, High Recall: {metrics['high_recall']}%)")

    # Verify train page content confirms 1 correction was included
    html = train_resp.content.decode('utf-8')
    assert "Corrections: <strong>1</strong>" in html or "1 manager corrections" in html or corrections_used == 1, "Train page does not show 1 correction!"
    print(f"  -> Confirmed: Train page and metrics.json report {corrections_used} correction(s) included!")

    print("\nAcceptance Check (f) PASSED successfully!")


if __name__ == '__main__':
    run_check_f()
