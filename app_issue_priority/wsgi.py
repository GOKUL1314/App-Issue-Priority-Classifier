"""
WSGI config for app_issue_priority project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app_issue_priority.settings')

application = get_wsgi_application()
