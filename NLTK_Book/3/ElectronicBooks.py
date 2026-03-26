# Código completo para Project Gutenberg
import nltk, re, pprint
from urllib import request
from nltk import word_tokenize

url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
print("Tipo:", type(raw))
print("Longitud:", len(raw))
print("Primeros 75 caracteres:", raw[:75])

# Tokenización
tokens = word_tokenize(raw)
print("\nTipo de tokens:", type(tokens))
print("Número de tokens:", len(tokens))
print("Primeros 10 tokens:", tokens[:10])

# Crear texto NLTK
text = nltk.Text(tokens)
print("\nTexto[1024:1062]:", text[1024:1062])
print("\nColocaciones:")
print(text.collocations())

# Limpieza de metadatos
start = raw.find("PART I")
end = raw.rfind("End of Project Gutenberg's Crime")
raw = raw[start:end]
print("\nÍndice de 'PART I' después de limpiar:", raw.find("PART I"))