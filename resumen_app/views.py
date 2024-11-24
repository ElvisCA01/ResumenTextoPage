import os
from django.conf import settings
from django.shortcuts import render
from .utils import procesar_entrada, resumen_nltk, generate_summary, obtener_texto_de_pdf, ajustar_resumen_por_tamano, obtener_texto_de_url, limpiar_texto, es_archivo_valido, obtener_texto_de_archivo
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
    Procesa una solicitud de resumen según el tipo de entrada y genera el resumen.
    """
    if request.method == 'POST':
        input_texto = request.POST.get('input_texto', '')
        input_archivo = request.FILES.get('input_archivo')
        input_url = request.POST.get('input_url', '')
        input_tamano = request.POST.get('input_tamano', 'medio')

        try:
            contenido = None
            
            # Procesar texto escrito directamente
            if input_texto:
                contenido = limpiar_texto(input_texto)
                resumen = resumen_nltk(contenido, input_tamano)
            
            # Procesar archivos cargados
            elif input_archivo:
                if not es_archivo_valido(input_archivo.name):
                    return render(request, 'resumen.html', {
                        'error': 'Tipo de archivo no soportado. Por favor, use: .txt, .pdf, .doc, .docx, .odt, o .rtf'
                    })
                contenido = limpiar_texto(obtener_texto_de_archivo(input_archivo))
                resumen = generate_summary(contenido)
            
            # Procesar URL
            elif input_url:
                contenido = limpiar_texto(obtener_texto_de_url(input_url))
                resumen = generate_summary(contenido)

            else:
                return render(request, 'resumen.html', {
                    'error': 'No se proporcionó contenido válido para resumir.'
                })

            # Ajustar el tamaño del resumen (opcional)
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
