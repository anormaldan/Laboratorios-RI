import string
import nltk
import re
import os

from nltk.stem.porter import PorterStemmer


# Descarga del paquete punkt_tab para tokenización.
nltk.download('punkt_tab')
# Descarga de WordNet para lematización.
nltk.download('wordnet')

# Función para parsear documentos
def parse_documents(input_file, output_file):

    # Parsea los documentos CACM y los guarda en formato: número | título | texto
    with open(input_file, 'r', encoding='utf8', errors="ignore") as f:
        content = f.read()

    # Dividir por documentos (.I marca el inicio de cada documento)
    docs = re.split(r'\.I ', content)

    results = []
    for doc in docs:
        if not doc.strip():
            continue

        lines = doc.strip().split('\n')

        # Extraer número de documento (primera línea)
        num_match = re.match(r'(\d+)', lines[0])
        if not num_match:
            continue
        num = num_match.group(1)

        # Extraer título (.T) - texto entre .T y siguiente campo
        title_match = re.search(r'\.T\n(.*?)(?=\n\.)', doc, re.DOTALL)
        titulo = ' '.join(title_match.group(1).split()) if title_match else ""

        # Extraer abstract/texto (.W) - texto entre .W y siguiente campo
        text_match = re.search(r'\.W\n(.*?)(?=\n\.)', doc, re.DOTALL)
        texto = ' '.join(text_match.group(1).split()) if text_match else ""

        results.append(f"{num} | {titulo} | {texto}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(results))

    print(f"Documentos parseados: {len(results)}. Guardado en {output_file}")


# Función para parsear consultas
def parse_queries(input_file, output_file):

    # Parsea las consultas CACM y las guarda en formato: número | título | texto
    with open(input_file, 'r', encoding='utf8', errors="ignore") as f:
        content = f.read()

    # Dividir por consultas (.I marca el inicio de cada consulta)
    queries = re.split(r'\.I ', content)

    results = []
    for query in queries:
        if not query.strip():
            continue

        lines = query.strip().split('\n')

        # Extraer número de consulta (primera línea)
        num_match = re.match(r'(\d+)', lines[0])
        if not num_match:
            continue
        num = num_match.group(1)

        # Extraer texto de la consulta (.W) - texto entre .W y siguiente campo
        text_match = re.search(r'\.W\n(.*?)(?=\n\.|$)', query, re.DOTALL)
        texto = ' '.join(text_match.group(1).split()) if text_match else ""

        # Usar parte del texto como título (primeras palabras)
        titulo = texto[:50] + "..." if len(texto) > 50 else texto

        results.append(f"{num} | {titulo} | {texto}")

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(results))

    print(f"Consultas parseadas: {len(results)}. Guardado en {output_file}")

# Carga stopwords
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f.readlines() if word.strip())


def preprocess_text(text, re_punc, porter, snowball, lemmatizer, stopwords):
    # Procesamiento de texto

    # Reemplazar guiones con espacios
    text = text.replace('-', ' ')

    # Tokenizar
    tokens = nltk.word_tokenize(text)

    # Eliminar puntuación
    stripped = [re_punc.sub('', w) for w in tokens]

    # Convertir a minúsculas y filtrar vacíos
    lowercase = [w.lower() for w in stripped if w.strip()]

    # Filtrar números (eliminar tokens que contengan cualquier dígito)
    no_numbers = [w for w in lowercase if not any(char.isdigit() for char in w)]

    # Eliminar stopwords
    no_stopwords = [w for w in no_numbers if w not in stopwords]

    # Porter Stemming
    porter_stemmed = [porter.stem(w) for w in no_stopwords]

    # Snowball Stemming
    snowball_stemmed = [snowball.stem(w) for w in no_stopwords]

    # WordNet Lemmatization
    lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in no_stopwords]

    return {
        'tokens': ' '.join(tokens),
        'stripped': ' '.join(stripped),
        'lowercase': ' '.join(lowercase),
        'no_numbers': ' '.join(no_numbers),
        'no_stopwords': ' '.join(no_stopwords),
        'porter': ' '.join(porter_stemmed),
        'snowball': ' '.join(snowball_stemmed),
        'lemmatized': ' '.join(lemmatized)
    }


def preprocess_file(input_file, output_prefix, stopwords_file=None):
    # Preprocesa un archivo con formato: número | título | texto
    # Genera múltiples archivos de salida con diferentes procesamientos.

    if stopwords_file is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(base_dir, 'resources', 'CACM', 'common_words')

    # Configuración de herramientas
    punctuation = string.punctuation + "'"
    re_punc = re.compile('[%s]' % re.escape(punctuation))
    porter = PorterStemmer()
    snowball = nltk.stem.SnowballStemmer('english')
    lemmatizer = nltk.WordNetLemmatizer()
    stopwords = load_stopwords(stopwords_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Listas para cada tipo de procesamiento
    results = {
        'tokens': [],
        'stripped': [],
        'lowercase': [],
        'no_numbers': [],
        'no_stopwords': [],
        'porter': [],
        'snowball': [],
        'lemmatized': []
    }

    for line in lines:
        parts = line.strip().split(' | ', 2)  # Máximo 3 partes
        if len(parts) >= 1:
            num = parts[0]
            titulo = parts[1] if len(parts) > 1 else ""
            texto = parts[2] if len(parts) > 2 else ""

            # Preprocesar título (puede estar vacío)
            titulo_proc = preprocess_text(titulo, re_punc, porter, snowball, lemmatizer, stopwords) if titulo else {k: "" for k in results.keys()}
            # Preprocesar texto (puede estar vacío)
            texto_proc = preprocess_text(texto, re_punc, porter, snowball, lemmatizer, stopwords) if texto else {k: "" for k in results.keys()}

            # Agregar a cada lista - SIEMPRE, incluso si está vacío
            for key in results.keys():
                results[key].append(f"{num} | {titulo_proc[key]} | {texto_proc[key]}")

    # Guardar cada tipo de procesamiento en un archivo separado
    for key, data in results.items():
        output_file = f"{output_prefix}_{key}.txt"
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data))
        print(f"Guardado: {output_file}")


def main():
    print("=" * 60)
    print("PROCESAMIENTO DE COLECCIÓN CACM")
    print("=" * 60)

    # Rutas de archivos (relativas al directorio del script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_input = os.path.join(base_dir, 'resources', 'CACM', 'cacm.all')
    queries_input = os.path.join(base_dir, 'resources', 'CACM', 'query.text')

    # Archivos de salida parseados
    docs_parsed = 'output/docs/documentos_parseados.txt'
    queries_parsed = 'output/queries/consultas_parseadas.txt'

    # PASO 1: PARSEAR ARCHIVOS
    print("\n[1] Parseando archivos originales ---")
    parse_documents(docs_input, docs_parsed)
    parse_queries(queries_input, queries_parsed)

    # PASO 2: PREPROCESAR DOCUMENTOS
    print("\n[2] Preprocesando documentos ---")
    preprocess_file(docs_parsed, 'output/docs/documentos')

    # PASO 3: PREPROCESAR CONSULTAS
    print("\n[3] Preprocesando consultas ---")
    preprocess_file(queries_parsed, 'output/queries/consultas')

    print("PROCESAMIENTO COMPLETADO")
    print("\nArchivos generados:")
    print("\n  output/docs/:")
    print("    - documentos_parseados.txt (formato: num | titulo | texto)")
    print("    - documentos_tokens.txt")
    print("    - documentos_stripped.txt (sin puntuación)")
    print("    - documentos_lowercase.txt (minúsculas)")
    print("    - documentos_no_numbers.txt (sin tokens que contengan números)")
    print("    - documentos_no_stopwords.txt (sin stopwords)")
    print("    - documentos_porter.txt (Porter Stemming)")
    print("    - documentos_snowball.txt (Snowball Stemming)")
    print("    - documentos_lemmatized.txt (WordNet Lemmatization)")
    print("\n  output/queries/:")
    print("    - consultas_parseadas.txt (formato: num | titulo | texto)")
    print("    - consultas_tokens.txt")
    print("    - consultas_stripped.txt (sin puntuación)")
    print("    - consultas_lowercase.txt (minúsculas)")
    print("    - consultas_no_numbers.txt (sin tokens que contengan números)")
    print("    - consultas_no_stopwords.txt (sin stopwords)")
    print("    - consultas_porter.txt (Porter Stemming)")
    print("    - consultas_snowball.txt (Snowball Stemming)")
    print("    - consultas_lemmatized.txt (WordNet Lemmatization)")


if __name__ == '__main__':
    main()