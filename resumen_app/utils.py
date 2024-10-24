'''
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
import requests
from bs4 import BeautifulSoup
import re 
import fitz  # PyMuPDF

# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def limpiar_texto(texto):
    # Eliminar referencias entre corchetes y números
    texto = re.sub(r'\[\d+\]', '', texto)  # Eliminar referencias como [1]
    texto = re.sub(r'\b\d+\b', '', texto)  # Eliminar números solitarios
    texto = re.sub(r'\s+', ' ', texto)  # Reemplazar múltiples espacios por uno solo
    return texto.strip()

def resumen_nltk(texto):
    # Limpiar el texto antes de procesarlo
    texto_limpio = limpiar_texto(texto)
    
    # Tokenización de oraciones
    sentences = sent_tokenize(texto_limpio)
    stop_words = set(stopwords.words('spanish'))

    # Normalización y eliminación de palabras de parada
    words = word_tokenize(texto_limpio.lower())

    # Filtrar palabras
    words = [
        word for word in words 
        if word.isalnum() and word not in stop_words
    ]

    # Conteo de frecuencia de palabras
    frequency = defaultdict(int)
    for word in words:
        frequency[word] += 1

    # Puntuación de oraciones
    sentence_score = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                sentence_score[sentence] += frequency[word]

    # Ajustar el tamaño del resumen (puedes cambiar el umbral)
    if len(sentences) > 5:
        top_sentences = sorted(sentence_score, key=sentence_score.get, reverse=True)[:int(len(sentences) * 0.3)]
    else:
        top_sentences = sorted(sentence_score, key=sentence_score.get, reverse=True)

    return ' '.join(sorted(top_sentences, key=lambda x: sentences.index(x)))  # Mantener el orden original

# Funciones para manejar diferentes tipos de entrada
def obtener_texto_de_archivo(archivo):
    if archivo.name.endswith('.pdf'):
        return obtener_texto_de_pdf(archivo)
    else:
        return archivo.read().decode('utf-8')

def obtener_texto_de_pdf(archivo):
    # Leer el PDF usando PyMuPDF
    texto = ""
    pdf_document = fitz.open(stream=archivo.read(), filetype="pdf")  # Cambia aquí
    for page in pdf_document:
        texto += page.get_text()
    pdf_document.close()
    return texto

def obtener_texto_de_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Lanza un error si la respuesta no es 200
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extraer el texto dentro de las etiquetas <p>
    paragraphs = soup.find_all('p')
    texto = ' '.join([para.get_text() for para in paragraphs])
    return texto

def procesar_entrada(input_type, contenido):
    if input_type == 'texto':
        return contenido
    elif input_type == 'archivo':
        return obtener_texto_de_archivo(contenido)
    elif input_type == 'url':
        return obtener_texto_de_url(contenido)
'''

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
import requests
from bs4 import BeautifulSoup
import re
import fitz  # PyMuPDF
from transformers import pipeline  # Importar el pipeline de Hugging Face

# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Inicializar el modelo de resumen
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def limpiar_texto(texto):
    # Eliminar referencias entre corchetes y números
    texto = re.sub(r'\[\d+\]', '', texto)  # Eliminar referencias como [1]
    texto = re.sub(r'\b\d+\b', '', texto)  # Eliminar números solitarios
    texto = re.sub(r'\s+', ' ', texto)  # Reemplazar múltiples espacios por uno solo
    return texto.strip()

def resumen_nltk(texto):
    # Limpiar el texto antes de procesarlo
    texto_limpio = limpiar_texto(texto)
    
    # Tokenización de oraciones
    sentences = sent_tokenize(texto_limpio)
    stop_words = set(stopwords.words('spanish'))

    # Normalización y eliminación de palabras de parada
    words = word_tokenize(texto_limpio.lower())

    # Filtrar palabras
    words = [
        word for word in words 
        if word.isalnum() and word not in stop_words
    ]

    # Conteo de frecuencia de palabras
    frequency = defaultdict(int)
    for word in words:
        frequency[word] += 1

    # Puntuación de oraciones
    sentence_score = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                sentence_score[sentence] += frequency[word]

    # Ajustar el tamaño del resumen
    if len(sentences) > 5:
        top_sentences = sorted(sentence_score, key=sentence_score.get, reverse=True)[:int(len(sentences) * 0.3)]
    else:
        top_sentences = sorted(sentence_score, key=sentence_score.get, reverse=True)

    return ' '.join(sorted(top_sentences, key=lambda x: sentences.index(x)))  # Mantener el orden original

def resumen_transformers(texto):
    # Limpiar el texto antes de resumir
    texto_limpio = limpiar_texto(texto)
    
    # Dividir el texto si es demasiado largo (para evitar errores)
    max_length = 512  # Longitud máxima para el modelo BART
    texto_chunks = [texto_limpio[i:i + max_length] for i in range(0, len(texto_limpio), max_length)]
    
    resumen = []
    for chunk in texto_chunks:
        # Generar resumen
        resumen_chunk = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        resumen.append(resumen_chunk[0]['summary_text'])
    
    # Unir el resumen y limitarlo a 250 palabras
    resumen_final = ' '.join(resumen)
    return limitar_a_250_palabras(resumen_final)

def limitar_a_250_palabras(texto):
    palabras = texto.split()
    if len(palabras) > 250:
        return ' '.join(palabras[:250]) + '...'  # Limitar a 250 palabras y añadir "..."
    return texto

def obtener_texto_de_archivo(archivo):
    if archivo.name.endswith('.pdf'):
        return obtener_texto_de_pdf(archivo)
    else:
        return archivo.read().decode('utf-8')

def obtener_texto_de_pdf(archivo):
    # Leer el PDF usando PyMuPDF
    texto = ""
    pdf_document = fitz.open(stream=archivo.read(), filetype="pdf")  # Cambia aquí
    for page in pdf_document:
        texto += page.get_text()
    pdf_document.close()
    
    # Eliminar carátulas o encabezados
    return limpiar_caratulas(texto)

def limpiar_caratulas(texto):
    # Limpiar las carátulas, índices y encabezados
    # Eliminar líneas que parecen carátulas o índices (puedes ajustar los patrones según tus necesidades)
    # Ejemplo: Eliminar títulos en mayúsculas, líneas que comienzan con números, o que contienen palabras como "Índice"
    texto = re.sub(r'^[A-Z\s]+$', '', texto, flags=re.MULTILINE)  # Eliminar líneas en mayúsculas
    texto = re.sub(r'^\d+.*$', '', texto, flags=re.MULTILINE)  # Eliminar líneas que empiezan con números
    texto = re.sub(r'Índice|Contenido|Tabla de Contenidos', '', texto, flags=re.IGNORECASE)  # Eliminar líneas de índices
    texto = re.sub(r'^[^\n]*\n?', '', texto)  # Eliminar la primera línea
    texto = re.sub(r'^[^\n]*\n{1,}', '', texto)  # Eliminar una línea en blanco al inicio si existe
    texto = re.sub(r'\n{2,}', '\n', texto)  # Eliminar saltos de línea adicionales
    return texto.strip()

def obtener_texto_de_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Lanza un error si la respuesta no es 200
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extraer el texto dentro de las etiquetas <p>
    paragraphs = soup.find_all('p')
    texto = ' '.join([para.get_text() for para in paragraphs])
    return texto

def procesar_entrada(input_type, contenido):
    if input_type == 'texto':
        return contenido
    elif input_type == 'archivo':
        return obtener_texto_de_archivo(contenido)
    elif input_type == 'url':
        return obtener_texto_de_url(contenido)






