"""
ASGI config for app_issue_priority project.
"""

import os

try:
    from django.core.asgi import get_asgi_application
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app_issue_priority.settings')
    application = get_asgi_application()
except ImportError:
    # Older Django versions fallback
    application = None
