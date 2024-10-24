'''
from django.shortcuts import render
from .utils import procesar_entrada, resumen_nltk

def index(request):
    return render(request, 'index.html')

def resumir(request):
    if request.method == 'POST':
        input_texto = request.POST.get('input_texto', None)
        input_archivo = request.FILES.get('input_archivo', None)
        input_url = request.POST.get('input_url', None)
        input_metodo = request.POST.get('input_metodo', 'nltk')  # NLTK es el valor por defecto

        contenido = None

        # Procesar el contenido de acuerdo al tipo de input
        if input_texto:
            contenido = procesar_entrada('texto', input_texto)
        elif input_archivo:
            contenido = procesar_entrada('archivo', input_archivo)
        elif input_url:
            contenido = procesar_entrada('url', input_url)

        if contenido:
            if input_metodo == 'nltk':
                resumen = resumen_nltk(contenido)
            else:
                resumen = "No se proporcionó ningún contenido."

        return render(request, 'resumen.html', {'resumen': resumen})

    return render(request, 'resumen.html')
'''

from django.shortcuts import render
from .utils import procesar_entrada, resumen_transformers, resumen_nltk

def index(request):
    return render(request, 'index.html')

def resumir(request):
    if request.method == 'POST':
        input_texto = request.POST.get('input_texto', None)
        input_archivo = request.FILES.get('input_archivo', None)
        input_url = request.POST.get('input_url', None)
        input_metodo = request.POST.get('input_metodo', 'nltk')  # NLTK es el valor por defecto

        contenido = None

        # Procesar el contenido de acuerdo al tipo de input
        if input_texto:
            contenido = procesar_entrada('texto', input_texto)
        elif input_archivo:
            contenido = procesar_entrada('archivo', input_archivo)
        elif input_url:
            contenido = procesar_entrada('url', input_url)

        if contenido:
            if input_metodo == 'nltk':
                resumen = resumen_nltk(contenido)
            else:
                resumen = resumen_transformers(contenido)  # Usar el resumen de transformers para PDFs
        else:
            resumen = "No se proporcionó ningún contenido."

        return render(request, 'resumen.html', {'resumen': resumen})

    return render(request, 'resumen.html')

