import os
import csv
import json
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
from django.utils import timezone
from django.db.models import Case, When, Value, IntegerField, Count, Q
import pandas as pd

from Creator_Portal.models import AppIssue
from ml.constants import APP_AREAS, PRIORITIES
from ml.train import train_all_models, METRICS_PATH, MODEL_PATH
from ml.predict import PriorityPredictor

STATUS_LIST = ['Open', 'In Review', 'In Progress', 'Resolved']


def _get_manager_credentials():
    """Reads manager credentials from environment or defaults only if DEBUG is True."""
    user = os.environ.get('MANAGER_USER')
    password = os.environ.get('MANAGER_PASS')
    if not user and settings.DEBUG:
        user = "admin"
    if not password and settings.DEBUG:
        password = "admin123"
    return user, password


def manager_login(request):
    """Care Manager login view."""
    if request.session.get('manager_logged_in'):
        return redirect('manager_issues')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        valid_user, valid_pass = _get_manager_credentials()

        if valid_user and valid_pass and username == valid_user and password == valid_pass:
            request.session['manager_logged_in'] = True
            request.session['manager_user'] = username
            messages.success(request, f"Welcome to Care Manager Portal, {username}!")
            return redirect('manager_issues')
        else:
            messages.error(request, "Invalid Care Manager credentials.")

    return render(request, 'Manager/login.html')


def manager_logout(request):
    """Care Manager logout view."""
    request.session.flush()
    messages.info(request, "Manager session closed.")
    return redirect('manager_login')


def _is_manager_authenticated(request):
    return request.session.get('manager_logged_in', False)


def _format_waiting_time(created_at, resolved_at=None):
    """Formats waiting duration like '3h 20m' or '2d 4h'."""
    if not created_at:
        return "0m"
    end = resolved_at if resolved_at else timezone.now()
    diff = end - created_at
    total_seconds = max(0, int(diff.total_seconds()))
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def _filter_issues(request):
    """
    Shared filter helper for list view and CSV export:
    - Hides Resolved unless specifically requested in the status filter
    - Applies priority, area, status, and text search filters
    - Orders by: High (1) -> Medium (2) -> Low (3), then created_at ASCENDING (oldest waiting first)
    """
    queryset = AppIssue.objects.all().select_related('creator')

    selected_priority = request.GET.get('priority', '').strip()
    selected_area = request.GET.get('app_area', '').strip()
    selected_status = request.GET.get('status', '').strip()
    search_query = request.GET.get('q', '').strip()

    # Hide Resolved unless the status filter specifically asks for it
    if selected_status == 'Resolved':
        queryset = queryset.filter(status='Resolved')
    elif selected_status:
        queryset = queryset.filter(status=selected_status)
    else:
        queryset = queryset.exclude(status='Resolved')

    if selected_priority and selected_priority in PRIORITIES:
        queryset = queryset.filter(final_priority=selected_priority)

    if selected_area and selected_area in APP_AREAS:
        queryset = queryset.filter(app_area=selected_area)

    if search_query:
        queryset = queryset.filter(
            Q(title__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(product_name__icontains=search_query) |
            Q(creator__username__icontains=search_query) |
            Q(creator__full_name__icontains=search_query)
        )

    # Order: High (1), Medium (2), Low (3), then created_at ASCENDING (longest waiting first)
    priority_order = Case(
        When(final_priority='High', then=Value(1)),
        When(final_priority='Medium', then=Value(2)),
        When(final_priority='Low', then=Value(3)),
        default=Value(4),
        output_field=IntegerField()
    )
    ordered_queryset = queryset.annotate(priority_weight=priority_order).order_by('priority_weight', 'created_at')

    return ordered_queryset, selected_priority, selected_area, selected_status, search_query


def issues_list_view(request):
    """
    Care Manager Prioritized Issue Queue view:
    - Orders by priority weight (High -> Medium -> Low), then created_at ASCENDING
    - Shows Waiting column, link status, contacts shared, prediction with confidence, priority_source and rule_name
    """
    if not _is_manager_authenticated(request):
        messages.warning(request, "Please log in to access Manager tools.")
        return redirect('manager_login')

    issues, selected_priority, selected_area, selected_status, search_query = _filter_issues(request)

    # Calculate waiting time for each issue
    issues_list = list(issues)
    for issue in issues_list:
        issue.waiting_time = _format_waiting_time(issue.created_at, issue.resolved_at if issue.status == 'Resolved' else None)

    # Summary counts
    total_count = AppIssue.objects.count()
    high_count = AppIssue.objects.filter(final_priority='High').exclude(status='Resolved').count()
    medium_count = AppIssue.objects.filter(final_priority='Medium').exclude(status='Resolved').count()
    low_count = AppIssue.objects.filter(final_priority='Low').exclude(status='Resolved').count()
    open_count = AppIssue.objects.filter(status='Open').count()

    context = {
        'issues': issues_list,
        'app_areas': APP_AREAS,
        'priorities': PRIORITIES,
        'statuses': STATUS_LIST,
        'selected_priority': selected_priority,
        'selected_area': selected_area,
        'selected_status': selected_status,
        'search_query': search_query,
        'total_count': total_count,
        'high_count': high_count,
        'medium_count': medium_count,
        'low_count': low_count,
        'open_count': open_count,
    }
    return render(request, 'Manager/issues_list.html', context)


def update_issue_status(request, issue_id):
    """
    Updates status and final_priority for an issue:
    - Accepts status and final_priority only from fixed lists
    - If final_priority changes: sets priority_source='manager', corrected_at=now
    - Never modifies predicted_priority
    - Always redirects to manager_issues
    """
    if not _is_manager_authenticated(request):
        return redirect('manager_login')

    if request.method == 'POST':
        issue = get_object_or_404(AppIssue, id=issue_id)
        new_status = request.POST.get('status', issue.status).strip()
        new_priority = request.POST.get('final_priority', issue.final_priority).strip()
        manager_notes = request.POST.get('manager_notes', issue.manager_notes).strip()

        if new_status in STATUS_LIST:
            if new_status == 'Resolved' and issue.status != 'Resolved':
                issue.resolved_at = timezone.now()
            elif new_status != 'Resolved':
                issue.resolved_at = None
            issue.status = new_status

        if new_priority in PRIORITIES:
            if new_priority != issue.final_priority:
                issue.final_priority = new_priority
                issue.priority_source = "manager"
                issue.corrected_at = timezone.now()

        issue.manager_notes = manager_notes
        issue.save()

        messages.success(request, f"Updated issue #{issue.id} ('{issue.title}').")

    return redirect('manager_issues')


def _sanitize_csv_cell(value):
    """Protects against CSV formula injection by prefixing formula triggers with an apostrophe."""
    if value is None:
        return ""
    val_str = str(value)
    if val_str and val_str[0] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + val_str
    return val_str


def export_issues_csv(request):
    """
    Exports filtered prioritized issues into a CSV spreadsheet with injection protection.
    """
    if not _is_manager_authenticated(request):
        return redirect('manager_login')

    issues, _, _, _, _ = _filter_issues(request)

    response = HttpResponse(content_type='text/csv; charset=utf-8')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    response['Content-Disposition'] = f'attachment; filename="app_issues_prioritized_{timestamp}.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Issue ID',
        'Title',
        'Product Name',
        'App Area',
        'Final Priority',
        'ML Predicted Priority',
        'Confidence (%)',
        'Priority Source',
        'Rule Name',
        'Link Status',
        'Contacts Shared',
        'Hours Since Link Created',
        'Device',
        'Issues Last 30d',
        'Status',
        'Creator Username',
        'Creator Name',
        'Description',
        'Manager Notes',
        'Waiting Time',
        'Created At (UTC)',
        'Last Updated (UTC)',
        'Corrected At (UTC)',
        'Resolved At (UTC)'
    ])

    for issue in issues:
        waiting = _format_waiting_time(issue.created_at, issue.resolved_at if issue.status == 'Resolved' else None)
        writer.writerow([
            _sanitize_csv_cell(issue.id),
            _sanitize_csv_cell(issue.title),
            _sanitize_csv_cell(issue.product_name),
            _sanitize_csv_cell(issue.app_area),
            _sanitize_csv_cell(issue.final_priority),
            _sanitize_csv_cell(issue.predicted_priority),
            _sanitize_csv_cell(issue.confidence),
            _sanitize_csv_cell(issue.priority_source),
            _sanitize_csv_cell(issue.rule_name),
            _sanitize_csv_cell(issue.link_status),
            _sanitize_csv_cell(issue.contacts_shared),
            _sanitize_csv_cell(issue.hours_since_link_created),
            _sanitize_csv_cell(issue.device),
            _sanitize_csv_cell(issue.issues_last_30d),
            _sanitize_csv_cell(issue.status),
            _sanitize_csv_cell(issue.creator.username if issue.creator else 'Anonymous'),
            _sanitize_csv_cell(issue.creator.full_name if issue.creator else 'N/A'),
            _sanitize_csv_cell(issue.description.replace('\n', ' ')),
            _sanitize_csv_cell(issue.manager_notes.replace('\n', ' ')),
            _sanitize_csv_cell(waiting),
            _sanitize_csv_cell(issue.created_at.strftime('%Y-%m-%d %H:%M:%S') if issue.created_at else ''),
            _sanitize_csv_cell(issue.updated_at.strftime('%Y-%m-%d %H:%M:%S') if issue.updated_at else ''),
            _sanitize_csv_cell(issue.corrected_at.strftime('%Y-%m-%d %H:%M:%S') if issue.corrected_at else ''),
            _sanitize_csv_cell(issue.resolved_at.strftime('%Y-%m-%d %H:%M:%S') if issue.resolved_at else '')
        ])

    return response


def train_model_view(request):
    """
    Care Manager ML Training view:
    - Pulls manager corrections (priority_source == 'manager') as extra_df
    - Trains 5 algorithms and reloads the pipeline
    - Displays macro F1, High recall, per-class recall table, confusion matrix, baseline accuracy, rows used, corrections used
    - Shows visible warning if any model scores 100%
    """
    if not _is_manager_authenticated(request):
        messages.warning(request, "Please log in to access Manager tools.")
        return redirect('manager_login')

    training_done = False
    if request.method == 'POST':
        try:
            # Build DataFrame from AppIssue rows where priority_source == 'manager'
            manager_corrections = AppIssue.objects.filter(priority_source='manager')
            extra_df = None
            if manager_corrections.exists():
                correction_rows = []
                for row in manager_corrections:
                    correction_rows.append({
                        'description': row.description,
                        'app_area': row.app_area,
                        'link_status': row.link_status,
                        'contacts_shared': row.contacts_shared,
                        'hours_since_link_created': row.hours_since_link_created,
                        'device': row.device,
                        'issues_last_30d': row.issues_last_30d,
                        'priority': row.final_priority
                    })
                extra_df = pd.DataFrame(correction_rows)

            summary = train_all_models(extra_df=extra_df)
            PriorityPredictor.reload_model()
            training_done = True
            messages.success(
                request,
                f"Successfully trained 5 ML algorithms! Best model selected: {summary['best_algorithm']} "
                f"(F1 Score: {summary['best_f1_score']}%) with {summary['corrections_used']} manager corrections."
            )
        except Exception as e:
            messages.error(request, f"Error during model training: {str(e)}")

    metrics = None
    has_perfect_score = False
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                for name, stats in metrics.get('models', {}).items():
                    if stats.get('accuracy', 0) >= 100.0 or stats.get('f1_score', 0) >= 100.0:
                        has_perfect_score = True
                        break
        except Exception:
            metrics = None

    context = {
        'metrics': metrics,
        'model_exists': os.path.exists(MODEL_PATH),
        'training_done': training_done,
        'has_perfect_score': has_perfect_score,
    }
    return render(request, 'Manager/train_model.html', context)


def charts_view(request):
    """
    Care Manager Analytics & Charts view:
    - Uses .values().annotate(Count) for database aggregations
    - Manager agreement percentage: (final_priority == predicted_priority) / total
    - ML algorithms comparison chart
    """
    if not _is_manager_authenticated(request):
        messages.warning(request, "Please log in to access Manager tools.")
        return redirect('manager_login')

    total_issues = AppIssue.objects.count()

    # Priority counts via annotate
    priority_qs = AppIssue.objects.values('final_priority').annotate(count=Count('id'))
    priority_dict = {row['final_priority']: row['count'] for row in priority_qs}
    priority_counts = [priority_dict.get(p, 0) for p in PRIORITIES]

    # App area counts via annotate
    area_qs = AppIssue.objects.values('app_area').annotate(count=Count('id'))
    area_dict = {row['app_area']: row['count'] for row in area_qs}
    area_labels = APP_AREAS
    area_counts = [area_dict.get(a, 0) for a in APP_AREAS]

    # Status counts via annotate
    status_qs = AppIssue.objects.values('status').annotate(count=Count('id'))
    status_dict = {row['status']: row['count'] for row in status_qs}
    status_counts = [status_dict.get(s, 0) for s in STATUS_LIST]

    # Manager agreement figure
    agreement_count = AppIssue.objects.filter(final_priority=models.F('predicted_priority')).count() if hasattr(models, 'F') else 0
    # Let's count properly via Q or python query:
    agreement_count = AppIssue.objects.filter(
        Q(final_priority='High', predicted_priority='High') |
        Q(final_priority='Medium', predicted_priority='Medium') |
        Q(final_priority='Low', predicted_priority='Low')
    ).count()

    agreement_pct = round((agreement_count / total_issues) * 100, 1) if total_issues > 0 else 100.0

    # ML Algorithms benchmark data
    model_names = []
    accuracies = []
    f1_scores = []
    best_algo = "N/A"

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                m = json.load(f)
                best_algo = m.get('best_algorithm', 'N/A')
                for name, stats in m.get('models', {}).items():
                    model_names.append(name)
                    accuracies.append(stats.get('accuracy', 0))
                    f1_scores.append(stats.get('f1_score', 0))
        except Exception:
            pass

    context = {
        'total_issues': total_issues,
        'agreement_pct': agreement_pct,
        'agreement_count': agreement_count,
        'priority_labels_json': json.dumps(PRIORITIES),
        'priority_data_json': json.dumps(priority_counts),
        'area_labels_json': json.dumps(area_labels),
        'area_data_json': json.dumps(area_counts),
        'status_labels_json': json.dumps(STATUS_LIST),
        'status_data_json': json.dumps(status_counts),
        'model_names_json': json.dumps(model_names),
        'accuracies_json': json.dumps(accuracies),
        'f1_scores_json': json.dumps(f1_scores),
        'best_algo': best_algo,
    }
    return render(request, 'Manager/charts.html', context)
