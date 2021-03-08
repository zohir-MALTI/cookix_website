from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import auth


# Create your views here.
def signup(request):
    if request.method == 'POST':
        if request.POST["password1"] == request.POST["password2"]:
            try:
                user = User.objects.get(username=request.POST["username"])
                return render(request, 'accounts/signup.html', {'error': 'Le nom d\'utilisateur est déja pris'})
            except User.DoesNotExist:
                user = User.objects.create_user(request.POST["username"], password=request.POST["password1"])
                auth.login(request, user)
                return redirect('home')
        else:
            return render(request, 'accounts/signup.html', {'error': 'Les deux mots de passe ne sont pas identiques !'})
    else:
        return render(request, 'accounts/signup.html')

def login(request):
    if request.method == 'POST':
        user = auth.authenticate(username=request.POST["username"], password=request.POST["password"])
        print("111")
        if user is not None:
            print("heey")
            auth.login(request, user)
            return redirect('home')
        else:
            print("errorrrr")
            return render(request, 'accounts/login.html', {'error': 'Le nom d\'utilisateur et le mot de passe ne concordent pas !'})
    else:
        return render(request, 'accounts/login.html')

def logout(request):
    # TODO need to route homapage
    if request.method == 'POST':
        auth.logout(request)
        return redirect('home')

