import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
import requests
from bs4 import BeautifulSoup
import re
import fitz  # PyMuPDF
import fitz  # PyMuPDF
import torch
from transformers import pipeline


# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Verificar si hay GPU disponible
device = 0 if torch.cuda.is_available() else -1

# Inicializar el pipeline de summarization
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    summarizer = None

def generate_summary(text, max_length=100, min_length=50):
    """
    Genera un resumen utilizando el pipeline de transformers.
    """
    try:
        if summarizer is None:
            return "Error: El modelo no se pudo cargar correctamente."

        # Dividir el texto en chunks si es muy largo
        max_chunk_length = 1024
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 100:  # Solo procesar chunks con suficiente contenido
                summary = summarizer(chunk, 
                                   max_length=max_length, 
                                   min_length=min_length, 
                                   do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        return ' '.join(summaries)
    except Exception as e:
        return f"Error al generar el resumen: {str(e)}"


def ajustar_resumen_por_tamano(resumen, tamano):
    """
    Ajusta el tamaño del resumen según la opción seleccionada.
    """
    if not resumen:
        return ""
        
    oraciones = sent_tokenize(resumen)
    
    if tamano == 'corto':
        num_oraciones = min(2, len(oraciones))
    elif tamano == 'medio':
        num_oraciones = min(5, len(oraciones))
    else:  # largo
        num_oraciones = min(10, len(oraciones))
    
    return ' '.join(oraciones[:num_oraciones])


def limpiar_texto(texto):
    """
    Limpia el texto eliminando referencias, números, múltiples espacios, etc.
    """
    if not texto:
        return ""
    texto = re.sub(r'\[\d+\]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def ajustar_resumen_por_tamano(resumen, tamano):
    """
    Ajusta el tamaño del resumen según la opción seleccionada.
    """
    if not resumen:
        return ""
        
    oraciones = sent_tokenize(resumen)
    
    if tamano == 'corto':
        num_oraciones = min(2, len(oraciones))
    elif tamano == 'medio':
        num_oraciones = min(5, len(oraciones))
    else:  # largo
        num_oraciones = min(10, len(oraciones))
    
    return ' '.join(oraciones[:num_oraciones])

def resumen_nltk(texto, tamano):
    """
    Genera un resumen simple utilizando NLTK y ajusta el tamaño según la opción seleccionada.
    """
    texto_limpio = limpiar_texto(texto)
    sentences = sent_tokenize(texto_limpio)
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in word_tokenize(texto_limpio.lower())
             if word.isalnum() and word not in stop_words]
    
    frequency = defaultdict(int)
    for word in words:
        frequency[word] += 1

    sentence_score = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                sentence_score[sentence] += frequency[word]

    top_sentences = sorted(sentence_score, key=sentence_score.get, reverse=True)[:int(len(sentences) * 0.3)]
    resumen = ' '.join(sorted(top_sentences, key=lambda x: sentences.index(x)))

    # Ajustamos el tamaño del resumen
    return ajustar_resumen_por_tamano(resumen, tamano)

def obtener_texto_de_pdf(archivo):
    """
    Extrae texto de un archivo PDF utilizando PyMuPDF.
    """
    try:
        texto = ""
        # Asegurarse de que estamos al inicio del archivo
        archivo.seek(0)
        
        # Leer el contenido del archivo
        pdf_bytes = archivo.read()
        
        # Abrir el PDF desde los bytes
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
            # Extraer texto de cada página
            for pagina in pdf_document:
                texto += pagina.get_text()
        
        return texto.strip()
    except Exception as e:
        return f"Error al procesar el PDF: {str(e)}"

def obtener_texto_de_url(url, max_length=2000):
    """
    Obtiene texto de una URL extrayendo contenido de etiquetas <p> y limita el tamaño.
    Se agrega un parámetro max_length para limitar la cantidad de texto que se usa para el resumen.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    texto = ' '.join([para.get_text() for para in soup.find_all('p')])
    
    # Limitar la cantidad de texto que se toma (max_length caracteres)
    texto_limited = texto[:max_length]  # Limita a max_length caracteres
    return texto_limited

def procesar_entrada(input_type, contenido):
    """
    Procesa entrada desde texto, archivo o URL.
    """
    if input_type == 'texto':
        return contenido
    elif input_type == 'archivo':
        return obtener_texto_de_pdf(contenido)  # Procesar PDF
    elif input_type == 'url':
        return obtener_texto_de_url(contenido)  # Procesar URL
