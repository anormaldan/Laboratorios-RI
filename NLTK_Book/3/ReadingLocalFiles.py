# Para archivos locales, se utiliza la función integrada open(). El parámetro 'rU' es útil para manejar diferentes convenciones de saltos de línea.

import nltk, os

# 1. LEER UN ARCHIVO LOCAL
print("Leyendo archivo local document.txt")
f = open('document.txt', 'r', encoding='utf-8')
raw = f.read()
print("\nContenido completo del archivo:")
print(raw)
f.close()

# 2. LEER ARCHIVO LÍNEA POR LÍNEA
print("\n" + "="*50)
print("Lectura línea por línea")
f = open('document.txt', 'r', encoding='utf-8')
for line in f:
    print(line.strip())
f.close()

# 3. LEER ARCHIVO DE UN CORPUS NLTK
print("\n" + "="*50)
print("Leyendo archivo del corpus Gutenberg")
path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
# Abrir y leer el archivo
raw_corpus = open(path, 'r', encoding='utf-8').read()
print("Primeros 500 caracteres de Moby Dick:")
print(raw_corpus[:500])