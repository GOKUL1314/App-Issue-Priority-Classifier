from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.creator_login, name='creator_login'),
    path('register/', views.creator_register, name='creator_register'),
    path('logout/', views.creator_logout, name='creator_logout'),
    path('customer-care/', views.customer_care, name='customer_care'),
]
