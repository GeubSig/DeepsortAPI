from django.urls import path
from stream import views


urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed1/', views.video_feed1, name='video_feed1'),
    path('video_feed2/', views.video_feed2, name='video_feed2'),
    path('days', views.days_data),
    path('times', views.times_data),
    path('current_people', views.current_people),
    path('timeline', views.timeline)
]
