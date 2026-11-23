import os
from datetime import timedelta
from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
from Creator_Portal.models import CreatorAccount, AppIssue
from ml.predict import score_issue


class IssuePriorityPlatformTests(TestCase):
    def setUp(self):
        # Set manager credentials in test environment
        os.environ['MANAGER_USER'] = 'admin'
        os.environ['MANAGER_PASS'] = 'admin123'

        self.client = Client()
        self.creator = CreatorAccount(
            username="creator_tester",
            email="creator_tester@example.com",
            full_name="Creator Tester",
            channel_or_app_name="Handmade Store"
        )
        self.creator.set_password("pass123")
        self.creator.save()

    def test_rules_never_lower_priority(self):
        """
        Decision check: Rules may only RAISE a priority, never lower it.
        Cosmetic downgrade rules are removed.
        """
        # Test cosmetic text: should NOT be downgraded by rules
        cosmetic_text = "Minor typo in store description: font contrast is slightly off"
        res = score_issue(
            description=cosmetic_text,
            app_area="Other",
            link_status="Draft",
            contacts_shared=0
        )
        # Even if rule is evaluated, it never lowers priority
        self.assertIn(res['final_priority'], ['High', 'Medium', 'Low'])
        self.assertNotEqual(res['priority_source'], 'cosmetic_downgrade')

        # Test broken link on Live with contacts shared: raises to High
        broken_link_res = score_issue(
            description="my friends say my link is not opening, 404 error",
            app_area="Product Link",
            link_status="Live",
            contacts_shared=15
        )
        self.assertEqual(broken_link_res['final_priority'], 'High')
        self.assertIn(broken_link_res['priority_source'], ['rule', 'model'])

        # Test account locked: raises to High
        account_res = score_issue(
            description="cannot log in, account locked and someone else in my account",
            app_area="Login & Account"
        )
        self.assertEqual(account_res['final_priority'], 'High')
        self.assertIn(account_res['priority_source'], ['rule', 'model'])

    def test_same_text_with_different_link_context_gives_different_priority(self):
        """
        Check context sensitivity: same text with different link context produces different priority.
        """
        text = "my friends say my link is not opening"
        # Context 1: Live link with 40 contacts shared
        res_live = score_issue(
            description=text,
            app_area="Product Link",
            link_status="Live",
            contacts_shared=40
        )

        # Context 2: Draft link with 0 contacts shared
        res_draft = score_issue(
            description=text,
            app_area="Product Link",
            link_status="Draft",
            contacts_shared=0
        )

        self.assertEqual(res_live['final_priority'], 'High')
        self.assertEqual(res_draft['final_priority'], 'Low')
        self.assertNotEqual(res_live['final_priority'], res_draft['final_priority'])

    def test_open_queue_order_and_resolved_hidden(self):
        """
        Manager queue order must be:
        High > Medium > Low, then created_at ASCENDING (longest waiting first).
        Resolved issues must be hidden unless explicitly requested.
        """
        now = timezone.now()

        # Older High issue (waiting 4 hours)
        h_old = AppIssue.objects.create(
            creator=self.creator,
            title="High Old",
            description="High priority issue older",
            app_area="Product Link",
            final_priority="High",
            predicted_priority="High",
            status="Open"
        )
        AppIssue.objects.filter(id=h_old.id).update(created_at=now - timedelta(hours=4))

        # Newer High issue (waiting 1 hour)
        h_new = AppIssue.objects.create(
            creator=self.creator,
            title="High New",
            description="High priority issue newer",
            app_area="Login & Account",
            final_priority="High",
            predicted_priority="High",
            status="Open"
        )
        AppIssue.objects.filter(id=h_new.id).update(created_at=now - timedelta(hours=1))

        # Medium issue (waiting 3 hours)
        m_issue = AppIssue.objects.create(
            creator=self.creator,
            title="Medium Issue",
            description="Medium priority issue",
            app_area="Image / Product Upload",
            final_priority="Medium",
            predicted_priority="Medium",
            status="Open"
        )
        AppIssue.objects.filter(id=m_issue.id).update(created_at=now - timedelta(hours=3))

        # Low issue (waiting 5 hours)
        l_issue = AppIssue.objects.create(
            creator=self.creator,
            title="Low Issue",
            description="Low priority issue",
            app_area="Other",
            final_priority="Low",
            predicted_priority="Low",
            status="Open"
        )
        AppIssue.objects.filter(id=l_issue.id).update(created_at=now - timedelta(hours=5))

        # Resolved issue
        resolved_issue = AppIssue.objects.create(
            creator=self.creator,
            title="Resolved Issue",
            description="Already resolved issue",
            app_area="Other",
            final_priority="High",
            predicted_priority="High",
            status="Resolved"
        )

        # Log in manager
        login_resp = self.client.post(reverse('manager_login'), {'username': 'admin', 'password': 'admin123'}, follow=True)
        self.assertEqual(login_resp.status_code, 200)

        response = self.client.get(reverse('manager_issues'))
        self.assertEqual(response.status_code, 200)

        issues_in_queue = list(response.context['issues'])
        issue_ids = [i.id for i in issues_in_queue]

        # 1. Resolved issue must be hidden by default
        self.assertNotIn(resolved_issue.id, issue_ids)

        # 2. Order must be: High Old (longest waiting) -> High New -> Medium -> Low
        expected_order = [h_old.id, h_new.id, m_issue.id, l_issue.id]
        self.assertEqual(issue_ids, expected_order)

        # 3. If filtered for Resolved, Resolved issue must show up
        resolved_response = self.client.get(reverse('manager_issues'), {'status': 'Resolved'})
        resolved_ids = [i.id for i in resolved_response.context['issues']]
        self.assertIn(resolved_issue.id, resolved_ids)
        self.assertNotIn(h_old.id, resolved_ids)

    def test_creator_pages_never_contain_priority_or_confidence(self):
        """
        Creators never see priority, confidence, or rule text.
        They see only: 'Complaint received. Ticket #N. Our care team will review it.' and table columns without priority/confidence.
        """
        # Log in creator
        self.client.post(reverse('creator_login'), {'username': 'creator_tester', 'password': 'pass123'})

        # Submit an issue
        response = self.client.post(reverse('customer_care'), {
            'app_area': 'Product Link',
            'description': 'my friends say my link is not opening, please fix',
            'product_name': 'My Ceramic Mug',
            'title': 'Link broken for friends',
            'link_status': 'Live',
            'contacts_shared': '30',
            'hours_since_link_created': '5',
            'device': 'Android',
            'issues_last_30d': '1'
        }, follow=True)

        self.assertEqual(response.status_code, 200)

        # Check flash message contains Complaint received Ticket #
        self.assertContains(response, "Complaint received. Ticket #")
        # Ensure message does NOT leak priority or rules
        self.assertNotContains(response, "Assigned Priority")
        self.assertNotContains(response, "Rule triggered")
        self.assertNotContains(response, "RULE_")

        # Check that table headings in Creator Customer Care page do not have Priority or Confidence
        html = response.content.decode('utf-8')
        # Table headers in customer_care.html: Ticket #, Title & Description, Area, Status, Date
        self.assertNotIn("<th scope=\"col\" class=\"text-center\">Priority</th>", html)
        self.assertNotIn("<th scope=\"col\">AI Confidence</th>", html)
        self.assertNotIn("preview-priority", html)

    def test_manager_correction_sets_source_and_keeps_predicted_priority(self):
        """
        A manager correction sets priority_source='manager' and keeps predicted_priority.
        """
        issue = AppIssue.objects.create(
            creator=self.creator,
            title="Style not changing",
            description="Color I picked is not applying in builder",
            app_area="Site Builder (styles)",
            predicted_priority="Medium",
            final_priority="Medium",
            confidence=85.0,
            priority_source="model",
            status="Open"
        )

        # Log in manager
        self.client.post(reverse('manager_login'), {'username': 'admin', 'password': 'admin123'}, follow=True)

        # Manager updates priority to High
        update_resp = self.client.post(
            reverse('update_issue_status', args=[issue.id]),
            {
                'status': 'In Progress',
                'final_priority': 'High',
                'manager_notes': 'Escalated because creator has active sale'
            },
            follow=True
        )
        self.assertEqual(update_resp.status_code, 200)

        issue.refresh_from_db()
        self.assertEqual(issue.final_priority, 'High')
        self.assertEqual(issue.predicted_priority, 'Medium')  # Never modified!
        self.assertEqual(issue.priority_source, 'manager')   # Marked as manager correction!
        self.assertIsNotNone(issue.corrected_at)             # Timestamp set!
        self.assertEqual(issue.status, 'In Progress')
        self.assertEqual(issue.manager_notes, 'Escalated because creator has active sale')

    def test_deleted_creator_session_handled_gracefully(self):
        """
        If session creator_id points to a deleted account, flush session and redirect to login instead of crashing.
        """
        session = self.client.session
        session['creator_id'] = 999999  # Non-existent ID
        session.save()

        response = self.client.get(reverse('customer_care'))
        self.assertRedirects(response, reverse('creator_login'))
        self.assertIsNone(self.client.session.get('creator_id'))
