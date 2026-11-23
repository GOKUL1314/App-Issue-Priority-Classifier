from django.db import models
from django.contrib.auth.hashers import make_password, check_password
from ml.constants import APP_AREAS, PRIORITIES, LINK_STATUSES, DEVICES

class CreatorAccount(models.Model):
    username = models.CharField(max_length=100, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=256)
    full_name = models.CharField(max_length=150)
    channel_or_app_name = models.CharField(max_length=150, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)

    def __str__(self):
        return f"{self.username} ({self.full_name})"

    class Meta:
        db_table = 'creator_accounts'
        verbose_name = 'Creator Account'
        verbose_name_plural = 'Creator Accounts'


class AppIssue(models.Model):
    PRIORITY_CHOICES = [(p, p) for p in PRIORITIES]
    APP_AREA_CHOICES = [(a, a) for a in APP_AREAS]
    LINK_STATUS_CHOICES = [(s, s) for s in LINK_STATUSES]
    DEVICE_CHOICES = [(d, d) for d in DEVICES]

    STATUS_CHOICES = [
        ('Open', 'Open'),
        ('In Review', 'In Review'),
        ('In Progress', 'In Progress'),
        ('Resolved', 'Resolved'),
    ]

    creator = models.ForeignKey(
        CreatorAccount,
        on_delete=models.CASCADE,
        related_name='issues',
        null=True,
        blank=True
    )
    title = models.CharField(max_length=255)
    description = models.TextField(max_length=2000)
    product_name = models.CharField(max_length=255, blank=True, default='')
    app_area = models.CharField(max_length=100, choices=APP_AREA_CHOICES)
    link_status = models.CharField(max_length=50, choices=LINK_STATUS_CHOICES, default='None')
    contacts_shared = models.IntegerField(default=0)
    hours_since_link_created = models.FloatField(default=0.0)
    device = models.CharField(max_length=50, choices=DEVICE_CHOICES, default='Desktop')
    issues_last_30d = models.IntegerField(default=0)

    predicted_priority = models.CharField(max_length=50, default='Medium')
    final_priority = models.CharField(max_length=50, choices=PRIORITY_CHOICES, default='Medium')
    confidence = models.FloatField(default=0.0)
    priority_source = models.CharField(max_length=50, default='model')  # model, rule, rules_only, manager
    rule_name = models.CharField(max_length=255, blank=True, default='')

    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='Open')
    manager_notes = models.TextField(blank=True, default='')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    corrected_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    def priority_badge_class(self):
        mapping = {
            'High': 'badge bg-warning text-dark',
            'Medium': 'badge bg-info text-dark',
            'Low': 'badge bg-secondary',
        }
        return mapping.get(self.final_priority, 'badge bg-secondary')

    def status_badge_class(self):
        mapping = {
            'Open': 'badge bg-danger',
            'In Review': 'badge bg-warning text-dark',
            'In Progress': 'badge bg-primary',
            'Resolved': 'badge bg-success',
        }
        return mapping.get(self.status, 'badge bg-secondary')

    def priority_rank(self):
        mapping = {'High': 1, 'Medium': 2, 'Low': 3}
        return mapping.get(self.final_priority, 4)

    def __str__(self):
        return f"[{self.final_priority}] {self.title} (by {self.creator.username if self.creator else 'Anonymous'})"

    class Meta:
        db_table = 'app_issues'
        verbose_name = 'App Issue'
        verbose_name_plural = 'App Issues'
        ordering = ['-created_at']
