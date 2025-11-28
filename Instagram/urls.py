
from django.contrib import admin
from django.urls import path ,include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('myapp.urls')),
<<<<<<< HEAD
]
=======
]
>>>>>>> bdd25032e861245c83f7acf2431752f74402aeb1
