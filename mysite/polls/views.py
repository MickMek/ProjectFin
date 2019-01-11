from django.shortcuts import render
from django.shortcuts import HttpResponse
from .models import Post
# Create your views here.

def home(request):
	return render(request, 'polls/home.html')

def projects(request):
	return render(request, 'polls/projects.html',{'title':'Projects'})

