import os
from django.conf import settings
from django.shortcuts import render
from .utils import procesar_entrada, resumen_nltk, generate_summary, obtener_texto_de_pdf, ajustar_resumen_por_tamano, obtener_texto_de_url, limpiar_texto
def ensure_media_directory():
    """
    Asegura que el directorio MEDIA_ROOT exista.
    """
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)


def index(request):
    """
    Renderiza la página de inicio principal.
    """
    return render(request, 'index.html')


def inicio(request):
    """
    Renderiza la página de inicio.
    """
    return render(request, 'inicio.html')


def resumir(request):
    """
    Procesa una solicitud de resumen según el tipo de entrada y el tamaño deseado.
    """
    if request.method == 'POST':
        input_texto = request.POST.get('input_texto', '')
        input_archivo = request.FILES.get('input_archivo')
        input_url = request.POST.get('input_url', '')
        input_metodo = request.POST.get('input_metodo', 'nltk')
        input_tamano = request.POST.get('input_tamano', 'medio')

        try:
            contenido = None
            
            # Determinar la fuente del contenido
            if input_archivo:
                contenido = obtener_texto_de_pdf(input_archivo)
            elif input_texto:
                contenido = input_texto
            elif input_url:
                contenido = obtener_texto_de_url(input_url)

            if not contenido:
                return render(request, 'resumen.html', {
                    'error': 'No se proporcionó contenido válido para resumir.'
                })

            # Limpiar el texto
            contenido = limpiar_texto(contenido)

            # Generar el resumen según el método seleccionado
            if input_metodo == 'transformers':
                resumen = generate_summary(contenido)
            else:  # nltk
                resumen = resumen_nltk(contenido, input_tamano)

            # Ajustar el tamaño del resumen
            resumen_final = ajustar_resumen_por_tamano(resumen, input_tamano)

            return render(request, 'resumen.html', {
                'resumen': resumen_final,
                'texto_original': contenido[:1000] + '...' if len(contenido) > 1000 else contenido
            })

        except Exception as e:
            return render(request, 'resumen.html', {
                'error': f'Error durante el procesamiento: {str(e)}'
            })

    return render(request, 'resumen.html')


def manejar_archivo(input_archivo):
    """
    Procesa un archivo proporcionado, asegurándose de manejar correctamente archivos PDF.
    """
    ensure_media_directory()  # Asegurarse de que MEDIA_ROOT existe

    # No es necesario guardar el archivo en el sistema de archivos, ya que Django maneja este paso
    safe_filename = os.path.basename(input_archivo.name)
    archivo_path = input_archivo

    # Si el archivo es PDF, lo procesamos de manera especial
    if safe_filename.endswith('.pdf'):
        # Pasamos el archivo a la función de procesamiento de PDF
        return generar_resumen_pdf(archivo_path)
    else:
        # Si el archivo no es PDF, procesamos como texto
        return procesar_entrada('archivo', input_archivo)

def generar_resumen_pdf(archivo):
    """
    Esta función procesa un archivo PDF cargado y genera el resumen utilizando el modelo BART.
    """
    texto_pdf = obtener_texto_de_pdf(archivo)  # Obtener texto del archivo PDF
    return generate_summary(texto_pdf)  # Generar resumen usando Hugging Face BART
