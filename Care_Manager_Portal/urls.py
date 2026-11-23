from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.manager_login, name='manager_login'),
    path('logout/', views.manager_logout, name='manager_logout'),
    path('train/', views.train_model_view, name='manager_train'),
    path('charts/', views.charts_view, name='manager_charts'),
    path('issues/', views.issues_list_view, name='manager_issues'),
    path('update-status/<int:issue_id>/', views.update_issue_status, name='update_issue_status'),
    path('export/', views.export_issues_csv, name='manager_export'),
]
