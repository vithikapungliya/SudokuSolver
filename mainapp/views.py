from django.shortcuts import render, redirect
from . import sudokusolver
from .models import Image
from django.contrib import messages


# Create your views here.
def home(request):
    if request.method == 'POST':
        upload = request.FILES['image']
        Image.objects.create(image=upload)


        images=Image.objects.all()
        # print(images[0].image.url)
        img = images[0]

        unsolved, solved = sudokusolver.call_solver(img.image.url)
        
        context={
            'img':img,
            'unsolved':unsolved, 
            'solved':solved
        }
        return render(request, 'sudoku_solver.html', context)


    Image.objects.all().delete()
    
    return render(request,'index.html')