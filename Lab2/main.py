import string
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Descarga del paquete punkt_tab para tokenización.
nltk.download('punkt_tab')
# Descargar las palabras vacías
nltk.download('stopwords')


def main():
    # Carga del archivo txt.
    with open('resources/2591-0.txt', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    # Tokenización y creación del objeto Text.
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)
    # Guardado del texto original en un archivo.
    with open('output/output_original.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(text)))
    # Imprime las primeras 100 palabras del texto.
    print(text[:100])
    print("Tokenizado completo. Guardado en output/output_original.txt")

    # Eliminacion de puntuación.

    # Puntuaction incluye el apóstrofe especial ’
    punctuation = string.punctuation + "’"
    # Deteccion de caracteres de puntuación usando expresiones regulares.
    re_punc = re.compile('[%s]' % re.escape(punctuation))
    # Eliminación de la puntuación de las palabras del texto.
    stripped = [re_punc.sub('', w) for w in text]
    # Guardado de las palabras sin puntuación en un archivo.
    with open('output/output_stripped.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(stripped)))
    # Imprime las primeras 100 palabras sin puntuación.
    print(stripped)
    print("Puntuación eliminada. Guardado en output/output_stripped.txt")

    # Conversión a minúsculas.

    # Conversión de palabras a minúsculas.
    # Stripped puede ser reemplazada por text si se
    # desea conservar la puntuación.
    lw_text = [word.lower() for word in stripped]
    # Guardado de las palabras en minúsculas en un archivo.
    with open('output/output_lowercase.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(lw_text)))
    # Imprime las primeras 100 palabras en minúsculas.
    print(lw_text)
    print("Convertido a minúsculas. Guardado en output/output_lowercase.txt")

    # Definir el idioma para identificar palabras vacias
    idioma = 'english'
    stop_words = set(stopwords.words(idioma))
    # Eliminar palabras  vacias
    filtered_text = [
        w for w in lw_text
        if w.isalpha() and w not in stop_words
    ]
    # Guardado sin palabras vacias
    with open('output/output_no_stopwords.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered_text))
    # Imprime las primeras 100 palabras
    print("\nPrimeras 100 palabras (sin stopwords):")
    print(filtered_text[:100])

    # Ejemplo con re.

    # Búsqueda de todas las palabras que contienen 'cat'.
    cat_words_re = re.compile(r'.*cat.*')
    # Filtrado de palabras que coinciden con la expresión regular
    # 'cat' dentro del array de minusculas.
    cat_words = [w for w in filtered_text if cat_words_re.match(w)]
    # Guardado de las palabras que contienen 'cat' en un archivo.
    with open('output/output_cats.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(cat_words)))
    # Imprime las primeras 100 palabras que contienen 'cat'.
    print(cat_words)
    print("Palabras que contienen 'cat' encontradas. Guardado en output/output_cats.txt")

    # Porter Stemming

    porter = PorterStemmer()
    porter_words = [porter.stem(w) for w in filtered_text]
    # Guardado de las palabras en un archivo.
    with open('output/output_porter.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(porter_words))
    # Imprime las primeras 100 palabras con Porter Stemmer.
    print("\nPrimeras 100 palabras con Porter Stemmer:")
    print(porter_words[:100])

    # Snowball Stemming
    snowball = SnowballStemmer("english")
    snowball_words = [snowball.stem(w) for w in filtered_text]
    # Guardado de las palabras en un archivo.
    with open('output/output_snowball.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(snowball_words))
    # Imprime las primeras 100 palabras con Porter Stemmer.
    print("\nPrimeras 100 palabras con Snowball Stemmer:")
    print(snowball_words[:100])

    # WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_text]
    # Guardado de las palabras en un archivo.
    with open('output/output_lemmatizer.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lemma_words))
    # Imprime las primeras 100 palabras con Porter Stemmer.
    print("\nPrimeras 100 palabras con WordNet Lemmatizer:")
    print(lemma_words[:100])

    print("\nPreprocesamiento completo terminado.")


if __name__ == '__main__':
    main()
