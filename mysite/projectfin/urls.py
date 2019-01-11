from django.conf.urls import include, url
from projectfin.views import goToSPMarket
from projectfin.views import HomeView
from projectfin.views import goToCryptoMarket
from projectfin.views import TemplateView
from . import views

urlpatterns = [
	url(r'^$', HomeView.get, name='projectfin'),
	url(r'^pred/$', views.pred, name='pred'),
	url(r'^cpred/$', views.cpred, name='cpred'),
    url(r'^plot/$', views.plot, name='plot'),
    url(r'^cplot/$', views.cplot, name='cplot'),
    url(r'^montecplot/$', views.montecplot, name='montecplot'),
    url(r'^viewplot/', TemplateView.as_view(template_name="temp-plot.html"),
                   name='viewplot'),
    url(r'^frontierplot/$', views.frontierplot, name='frontierplot'),
    url(r'^clusterplot/$', views.clusterplot, name='clusterplot'),
    url(r'^cfrontierplot/$', views.cfrontierplot, name='cfrontierplot'),
	url(r'^sp500/$', goToSPMarket.get , name='sp500'),
	url(r'^crypto/$', goToCryptoMarket.get, name='crypto'),
	]


    