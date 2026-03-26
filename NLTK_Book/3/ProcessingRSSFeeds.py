# Este código requiere la librería feedparser. Permite acceder a contenidos de blogs de manera estructurada.

import feedparser
from bs4 import BeautifulSoup
from nltk import word_tokenize
import nltk, re, pprint

# (Solo la primera vez)
# nltk.download('punkt')

# Parsear el feed RSS (Atom)
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")

# Título del feed
print("Título del feed:", llog['feed']['title'])

# Número de entradas en el feed
print("Número de entradas:", len(llog.entries))

# Seleccionar una entrada específica
post = llog.entries[2]

# Título del post
print("Título del post:", post.title)

# Contenido HTML del post
content = post.content[0].value

# Mostrar una parte del HTML
print("Primeros 70 caracteres del contenido:", content[:70])

# Limpiar HTML → texto plano
raw = BeautifulSoup(content, 'html.parser').get_text()

# Tokenización
tokens = word_tokenize(raw)

# Mostrar tokens
print("Primeros 20 tokens:", tokens[:20])