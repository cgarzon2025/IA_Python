import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import os

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def analyze_text(text):
    # Tokenización de oraciones
    sentences = sent_tokenize(text)
    
    # Tokenización de palabras
    words = word_tokenize(text.lower())
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Análisis de frecuencia
    fdist = FreqDist(filtered_words)
    
    return {
        'num_sentences': len(sentences),
        'num_words': len(filtered_words),
        'most_common_words': fdist.most_common(10),
        'sentences': sentences[:5]  # Primeras 5 oraciones como ejemplo
    }

def plot_word_frequency(fdist):
    plt.figure(figsize=(12, 6))
    fdist.plot(10, title='Distribución de las 10 palabras más comunes')
    plt.show()

if __name__ == "__main__":
    # Obtener el directorio actual del script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "book_aruba.pdf")
    
    print(f"Intentando abrir el archivo: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print("Texto extraído.\n")

    analysis = analyze_text(text)
    print(f"Número de oraciones: {analysis['num_sentences']}")
    print(f"Número de palabras (sin stopwords): {analysis['num_words']}")
    print("\nPalabras más comunes:")
    for word, freq in analysis['most_common_words']:
        print(f"{word}: {freq}")
    
    print("\nPrimeras 5 oraciones como ejemplo:")
    for i, sentence in enumerate(analysis['sentences'], 1):
        print(f"{i}. {sentence}")
    
    # Crear y mostrar el gráfico de frecuencia de palabras
    fdist = FreqDist([word for word in word_tokenize(text.lower()) if word.isalnum() and word not in stopwords.words('spanish')])
    plot_word_frequency(fdist) 