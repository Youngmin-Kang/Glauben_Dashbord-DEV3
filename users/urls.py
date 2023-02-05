from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("Glauben/", views.Glauben),
    path("login/", views.login_view, name = "éxito"),
    #path("GlaubenLogin/", views.GlaubenLogin_view),
    path("sidebarGlauben/", views.login_Sidebar, name = "sidebarGlauben"),
    path("FNsidebarGlauben/", views.fn_sidebar, name = "FNsidebarGlauben"),
    path("estadistica/", views.graficoBar),
    path("prediccion/",views.prediccion),
    path("prediccionFN/",views.prediccion_fn),
    path('GlaubenLogin/',views.register, name='login1'),
    path('GlaubenLogin/login/',views.glauben_login, name='login'),
    path('GlaubenLogin/register/',views.register, name='register'),
    path('logout/', views.glauben_logout, name='logout'),
    path('editarInfo/', views.editInfo, name='editarInfo'),
    path('<int:id>', views.view_user, name='view_user'),
    path('add/', views.add, name='add'),
    path('edit/<int:id>/', views.edit, name='edit'), 
    path('eliminar/<int:id>/', views.eliminar, name='eliminar'),
    path("multipleSDI/", views.multipleSDI_data, name = "multipleSDI1"), 
    
]
