from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from .models import CreatorAccount, AppIssue
from ml.constants import APP_AREAS, LINK_STATUSES, DEVICES
from ml.predict import score_issue


def creator_register(request):
    """Creator registration view."""
    if request.session.get('creator_id'):
        return redirect('customer_care')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '').strip()
        full_name = request.POST.get('full_name', '').strip()
        channel_name = request.POST.get('channel_name', '').strip()

        if not username or not email or not password or not full_name:
            messages.error(request, "Please fill in all required fields.")
            return render(request, 'Creator/register.html')

        if CreatorAccount.objects.filter(username=username).exists():
            messages.error(request, f"Username '{username}' is already taken.")
            return render(request, 'Creator/register.html')

        if CreatorAccount.objects.filter(email=email).exists():
            messages.error(request, f"Email '{email}' is already registered.")
            return render(request, 'Creator/register.html')

        creator = CreatorAccount(
            username=username,
            email=email,
            full_name=full_name,
            channel_or_app_name=channel_name
        )
        creator.set_password(password)
        creator.save()

        # Log in automatically
        request.session['creator_id'] = creator.id
        request.session['creator_username'] = creator.username
        request.session['creator_name'] = creator.full_name
        messages.success(request, f"Welcome {creator.full_name}! Your Creator account has been created.")
        return redirect('customer_care')

    return render(request, 'Creator/register.html')


def creator_login(request):
    """Creator login view."""
    if request.session.get('creator_id'):
        return redirect('customer_care')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        try:
            creator = CreatorAccount.objects.get(username=username)
            if creator.check_password(password):
                request.session['creator_id'] = creator.id
                request.session['creator_username'] = creator.username
                request.session['creator_name'] = creator.full_name
                messages.success(request, f"Welcome back, {creator.full_name}!")
                return redirect('customer_care')
            else:
                messages.error(request, "Invalid username or password.")
        except CreatorAccount.DoesNotExist:
            messages.error(request, "Account not found for that username.")

    return render(request, 'Creator/login.html')


def creator_logout(request):
    """Creator logout view."""
    request.session.flush()
    messages.info(request, "You have been logged out safely.")
    return redirect('creator_login')


def customer_care(request):
    """
    Creator Customer Care portal:
    - Submit issues (platform related issues only)
    - Validates inputs strictly
    - Score priority internally (never shown to creators)
    - Creators see only: Ticket #, title, area, status, date
    """
    creator_id = request.session.get('creator_id')
    if not creator_id:
        messages.warning(request, "Please log in to access the Creator Customer Care portal.")
        return redirect('creator_login')

    try:
        creator = CreatorAccount.objects.get(id=creator_id)
    except CreatorAccount.DoesNotExist:
        request.session.flush()
        messages.warning(request, "Account session expired or account removed. Please log in again.")
        return redirect('creator_login')

    demo_mode = getattr(settings, 'DEMO_MODE', True)

    if request.method == 'POST':
        raw_title = request.POST.get('title', '').strip()
        app_area = request.POST.get('app_area', '').strip()
        description = request.POST.get('description', '').strip()
        product_name = request.POST.get('product_name', '').strip()

        link_status = request.POST.get('link_status', 'None').strip()
        raw_contacts = request.POST.get('contacts_shared', '0').strip()
        raw_hours = request.POST.get('hours_since_link_created', '0').strip()
        device = request.POST.get('device', 'Desktop').strip()
        raw_issues_30d = request.POST.get('issues_last_30d', '0').strip()

        # Server-side validation
        errors = []
        if not description:
            errors.append("Description is required.")
        elif len(description) > 2000:
            errors.append("Description must be at most 2000 characters.")

        if app_area not in APP_AREAS:
            errors.append(f"Invalid app area '{app_area}'. Allowed: {', '.join(APP_AREAS)}.")

        if link_status not in LINK_STATUSES:
            errors.append(f"Invalid link status '{link_status}'. Allowed: {', '.join(LINK_STATUSES)}.")

        if device not in DEVICES:
            errors.append(f"Invalid device '{device}'. Allowed: {', '.join(DEVICES)}.")

        try:
            contacts_shared = int(raw_contacts)
            if contacts_shared < 0:
                errors.append("Contacts shared must be greater than or equal to 0.")
        except (ValueError, TypeError):
            errors.append("Contacts shared must be a valid non-negative integer.")
            contacts_shared = 0

        try:
            hours_since_link_created = float(raw_hours)
            if hours_since_link_created < 0:
                errors.append("Hours since link created must be greater than or equal to 0.")
        except (ValueError, TypeError):
            errors.append("Hours since link created must be a valid non-negative number.")
            hours_since_link_created = 0.0

        try:
            issues_last_30d = int(raw_issues_30d)
            if issues_last_30d < 0:
                errors.append("Issues last 30 days must be greater than or equal to 0.")
        except (ValueError, TypeError):
            errors.append("Issues last 30 days must be a valid non-negative integer.")
            issues_last_30d = 0

        if errors:
            for err in errors:
                messages.error(request, err)
        else:
            # Default title: first 60 chars of description if not provided
            title = raw_title if raw_title else description[:60]

            # Score priority internally (never shown to creator)
            score = score_issue(
                description=description,
                app_area=app_area,
                link_status=link_status,
                contacts_shared=contacts_shared,
                hours_since_link_created=hours_since_link_created,
                device=device,
                issues_last_30d=issues_last_30d
            )

            issue = AppIssue.objects.create(
                creator=creator,
                title=title,
                product_name=product_name,
                description=description,
                app_area=app_area,
                link_status=link_status,
                contacts_shared=contacts_shared,
                hours_since_link_created=hours_since_link_created,
                device=device,
                issues_last_30d=issues_last_30d,
                predicted_priority=score['ml_prediction'],
                final_priority=score['final_priority'],
                confidence=score['confidence'],
                priority_source=score['priority_source'],
                rule_name=score['rule_name'],
                status='Open'
            )

            # Creators see only: "Complaint received. Ticket #<id>. Our care team will review it."
            messages.success(request, f"Complaint received. Ticket #{issue.id}. Our care team will review it.")
            return redirect('customer_care')

    creator_issues = AppIssue.objects.filter(creator=creator).order_by('-created_at')

    context = {
        'creator': creator,
        'app_areas': APP_AREAS,
        'link_statuses': LINK_STATUSES,
        'devices': DEVICES,
        'demo_mode': demo_mode,
        'issues': creator_issues,
        'total_issues': creator_issues.count(),
        'open_issues': creator_issues.filter(status='Open').count(),
        'resolved_issues': creator_issues.filter(status='Resolved').count(),
    }
    return render(request, 'Creator/customer_care.html', context)
