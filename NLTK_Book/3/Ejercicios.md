# Ejercicios NLTK Book - Capítulo 3

## Ejemplos y ejercicios
```python
# Entrada
from urllib import request
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
type(raw)
len(raw)
raw[:75]
```

```python
# Salida
<class 'str'>
1135213
'*** START OF THE PROJECT GUTENBERG EBOOK 2554 ***\n\n\n\n\nCRIME AND PUNISHMENT\n'
```