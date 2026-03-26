#!/usr/bin/env python3
# Pequeño test para verificar el filtrado de números

import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Texto de prueba que contiene "10i"
test_text = "This contains 10i and other numbers like 123 and words like hello"

# Función de procesamiento actual
def preprocess_text(text):
    # Reemplazar guiones con espacios
    text = text.replace('-', ' ')
    print(f"Después de reemplazar guiones: {text}")

    # Tokenizar
    tokens = nltk.word_tokenize(text)
    print(f"Tokens: {tokens}")

    # Eliminar puntuación
    re_punc = re.compile('[%s]' % re.escape('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
    stripped = [re_punc.sub('', w) for w in tokens]
    print(f"Sin puntuación: {stripped}")

    # Convertir a minúsculas y filtrar vacíos
    lowercase = [w.lower() for w in stripped if w.strip()]
    print(f"Minúsculas: {lowercase}")

    # Filtrar números (eliminar tokens que contengan cualquier dígito)
    no_numbers = [w for w in lowercase if not any(char.isdigit() for char in w)]
    print(f"Sin números: {no_numbers}")

    return no_numbers

if __name__ == '__main__':
    nltk.download('punkt', quiet=True)
    result = preprocess_text(test_text)
    print(f"\nResultado final: {result}")

