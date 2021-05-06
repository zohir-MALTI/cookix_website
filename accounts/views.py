from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from .models import UserPreferences


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
        if user is not None:
            auth.login(request, user)
            return redirect('home')
        else:
            return render(request, 'accounts/login.html', {'error': 'Le nom d\'utilisateur et le mot de passe ne concordent pas !'})
    else:
        return render(request, 'accounts/login.html')


@login_required(login_url="/accounts/login")
def logout(request):
    # TODO need to route homapage
    if request.method == 'POST':
        auth.logout(request)
        return redirect('home')


# @login_required(login_url="/accounts/login")
def settings(request):
    user_settings = UserPreferences.objects.get(pk=request.user.id)
    if request.method == 'POST':
        print("seeeeeeeeetiings")
        user_settings.update(request.POST["vegetables"], request.POST["gluten"], request.POST["dairy"], request.POST["pork"],
                             request.POST["oven"], request.POST["microwave"], request.POST["blender"])
        return render(request, 'accounts/settings.html', {"success_msg": "Your changes have been updated successfully!"})
    else:
        return render(request, 'accounts/settings.html')

