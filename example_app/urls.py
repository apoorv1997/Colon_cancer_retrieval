from django.conf.urls import url, include
from django.contrib import admin
from example_app.views import ChatterBotAppView, ChatterBotApiView, MainWebsite,MLView


urlpatterns = [
    url(r'^index/app/', ChatterBotAppView.as_view(), name='main'),
    url(r'^index/$', MainWebsite.as_view(), name='main'),
    url(r'^admin/', admin.site.urls, name='admin'),
    url(r'^mlapi/',MLView.as_view() , name='mlapi'),
    url(r'^api/chatterbot/', include('chatterbot.ext.django_chatterbot.urls', namespace='chatterbot')),
]
