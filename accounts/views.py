from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import auth
from django.contrib.auth.decorators import login_required
# from .models import UserPreferences


def signup(request):
    if request.method == 'POST':
        username = request.POST["username"].lower()
        if request.POST["password1"] == request.POST["password2"]:
            try:
                new_user = User.objects.get(username=username)
                return render(request, 'accounts/signup.html', {'error': 'This username is already taken !'})
            except User.DoesNotExist:
                new_user = User.objects.create_user(username, password=request.POST["password1"],
                                                    email=request.POST["email"],
                                                    first_name = request.POST["firstname"],
                                                    last_name = request.POST["lastname"])
                new_user.save()
                auth.login(request, new_user)
                return redirect('home')
        else:
            return render(request, 'accounts/signup.html', {'error': 'Both of the passwords are not the same!'})
    else:
        return render(request, 'accounts/signup.html')


def login(request):
    if request.method == 'POST':
        username = request.POST["username"].lower()
        user = auth.authenticate(username=username, password=request.POST["password"])
        if user is not None:
            auth.login(request, user)
            return redirect('home')
        else:
            return render(request, 'accounts/login.html', {'error': 'The username and password do not match!'})
    else:
        return render(request, 'accounts/login.html')


@login_required(login_url="/accounts/login")
def logout(request):
    # TODO need to route homapage
    if request.method == 'POST':
        auth.logout(request)
        return redirect('home')


@login_required(login_url="/accounts/login")
def settings(request):
    user = User.objects.get(pk=request.user.id)
    print(user)
    if request.method == 'POST':
        username = request.POST["username"].lower()
        print("seeeeeeeeetiings")
        print("passs: ",request.POST["password"])
        user.username = username
        user.last_name = request.POST["lastname"]
        user.first_name = request.POST["firstname"]
        if request.POST["password"] != "":
            user.set_password(request.POST["password"])
        user.save()
        return render(request, 'accounts/settings.html', {"user": user,
                                                          "success_msg": "Your changes have been updated successfully!"})

    # if request.method == 'POST':
    #     print("seeeeeeeeetiings")
    #     user_settings.update(request.POST["vegetables"], request.POST["gluten"], request.POST["dairy"], request.POST["pork"],
    #                          request.POST["oven"], request.POST["microwave"], request.POST["blender"])
    # return render(request, 'accounts/settings.html', {"success_msg": "Your changes have been updated successfully!"})
    else:
        return render(request, 'accounts/settings.html', {"user": user})

