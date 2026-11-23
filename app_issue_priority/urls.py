"""
app_issue_priority URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render

def index_view(request):
    return render(request, 'index.html')

urlpatterns = [
    path('', index_view, name='home'),
    path('admin/', admin.site.urls),
    path('creator/', include('Creator_Portal.urls')),
    path('manager/', include('Care_Manager_Portal.urls')),
]
