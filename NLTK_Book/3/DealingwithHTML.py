# Para procesar páginas web, se descarga el HTML y se utiliza la librería BeautifulSoup para extraer solo el texto visible, eliminando etiquetas y scripts.

import nltk
from urllib import request
from bs4 import BeautifulSoup
from nltk import word_tokenize

# (Solo la primera vez)
# nltk.download('punkt')

# URL del artículo de la BBC
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
# Descarga del HTML completo
html = request.urlopen(url).read().decode('utf8')
# Mostrar una parte del HTML (opcional, solo para comprobar)
print("Primeros 60 caracteres del HTML:", html[:60])

# Limpieza del HTML → texto plano
raw = BeautifulSoup(html, 'html.parser').get_text()

# Tokenización
tokens = word_tokenize(raw)
tokens = tokens[110:390]
text = nltk.Text(tokens)
print("\nConcordancia para 'gene':")
text.concordance('gene')