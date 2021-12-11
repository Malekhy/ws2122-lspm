from django.shortcuts import render

def home(request):

    return render(request, "home.html")


def sampling(request):

    return render(request, "sampling.html")


