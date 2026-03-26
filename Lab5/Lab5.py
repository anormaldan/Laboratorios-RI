"""
Laboratorio 5: Recuperación de Información
Autores: Nava Campos Alejandro Dante, Ortega Pérez Andrea, y Tiro Cuanenemi Jonathan.

Objetivo:
Representar documentos y consultas utilizando el modelo vectorial con
esquemas de pesado TF y TF-IDF.

Entradas:
Lab3/output/docs/documentos_no_stopwords.txt
Lab3/output/queries/queries_no_stopwords.txt
Lab4/output/vocabulary/vocabularioReducido.txt

Salidas:
output/matrices/
    documentos_tf.txt
    queries_tf.txt
    documentos_tfidf.txt
    queries_tfidf.txt
"""

import os
import math
from collections import Counter


# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DOC_FILE = os.path.join(BASE_DIR, 'Lab3', 'output', 'docs',
                        'documentos_no_stopwords.txt')

QUERY_FILE = os.path.join(BASE_DIR, 'Lab3', 'output', 'queries',
                          'consultas_no_stopwords.txt')

VOCAB_FILE = os.path.join(BASE_DIR, 'Lab4', 'output', 'vocabulary',
                          'vocabularioReducido.txt')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'matrices')

DOC_TF_FILE = os.path.join(OUTPUT_DIR, 'documentos_tf.txt')
QUERY_TF_FILE = os.path.join(OUTPUT_DIR, 'consultas_tf.txt')

DOC_TFIDF_FILE = os.path.join(OUTPUT_DIR, 'documentos_tfidf.txt')
QUERY_TFIDF_FILE = os.path.join(OUTPUT_DIR, 'consultas_tfidf.txt')


# ---------------------------------------------------------
# Leer vocabulario
# ---------------------------------------------------------

def load_vocabulary(filepath):

    with open(filepath, encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]

    return vocab


# ---------------------------------------------------------
# Leer documentos
# ---------------------------------------------------------

def load_documents(filepath):

    docs = []

    with open(filepath, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(" | ", 2)

            if len(parts) != 3:
                continue

            _, titulo, texto = parts

            tokens = (titulo + " " + texto).split()

            docs.append(tokens)

    return docs


# ---------------------------------------------------------
# Leer queries
# ---------------------------------------------------------

def load_queries(filepath):

    queries = []

    with open(filepath, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(" | ", 2)

            if len(parts) < 2:
                continue

            # Formato: num | titulo | texto  (igual que documentos)
            titulo = parts[1] if len(parts) > 1 else ""
            texto  = parts[2] if len(parts) > 2 else ""

            tokens = (titulo + " " + texto).split()

            queries.append(tokens)

    return queries


# ---------------------------------------------------------
# Construir matriz TF
# ---------------------------------------------------------

def build_tf_matrix(collection, vocab):

    vocab_index = {term: i for i, term in enumerate(vocab)}

    matrix = []

    for tokens in collection:

        vector = [0] * len(vocab)

        freq = Counter(tokens)

        for term, count in freq.items():

            if term in vocab_index:

                vector[vocab_index[term]] = count

        matrix.append(vector)

    return matrix


# ---------------------------------------------------------
# Calcular DF
# ---------------------------------------------------------

def compute_df(collection, vocab):

    vocab_index = {term: i for i, term in enumerate(vocab)}

    df = [0] * len(vocab)

    for tokens in collection:

        seen = set(tokens)

        for term in seen:

            if term in vocab_index:

                df[vocab_index[term]] += 1

    return df


# ---------------------------------------------------------
# Calcular IDF
# ---------------------------------------------------------

def compute_idf(df, total_docs):

    idf = []

    for d in df:

        if d == 0:
            idf.append(0)
        else:
            idf.append(math.log(total_docs / d))

    return idf


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

def build_tfidf_matrix(tf_matrix, idf):

    matrix = []

    for row in tf_matrix:

        vector = []

        for tf, idf_val in zip(row, idf):

            vector.append(tf * idf_val)

        matrix.append(vector)

    return matrix


# ---------------------------------------------------------
# Guardar matriz
# ---------------------------------------------------------

def save_matrix(matrix, filepath):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:

        for row in matrix:

            f.write(" ".join(str(x) for x in row) + "\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("="*60)
    print("LABORATORIO 5: MODELO VECTORIAL")
    print("="*60)

    # --------------------------
    # Cargar datos
    # --------------------------

    vocab = load_vocabulary(VOCAB_FILE)

    print("Tamaño del vocabulario:", len(vocab))

    documents = load_documents(DOC_FILE)
    queries = load_queries(QUERY_FILE)

    print("Documentos:", len(documents))
    print("Queries:", len(queries))

    # --------------------------
    # TF documentos
    # --------------------------

    print("\nConstruyendo matriz TF de documentos...")

    doc_tf = build_tf_matrix(documents, vocab)

    save_matrix(doc_tf, DOC_TF_FILE)

    # --------------------------
    # TF queries
    # --------------------------

    print("Construyendo matriz TF de queries...")

    query_tf = build_tf_matrix(queries, vocab)

    save_matrix(query_tf, QUERY_TF_FILE)

    # --------------------------
    # IDF
    # --------------------------

    print("\nCalculando IDF...")

    df = compute_df(documents, vocab)

    idf = compute_idf(df, len(documents))

    # --------------------------
    # TF-IDF documentos
    # --------------------------

    print("Construyendo matriz TF-IDF documentos...")

    doc_tfidf = build_tfidf_matrix(doc_tf, idf)

    save_matrix(doc_tfidf, DOC_TFIDF_FILE)

    # --------------------------
    # TF-IDF queries
    # --------------------------

    print("Construyendo matriz TF-IDF queries...")

    query_tfidf = build_tfidf_matrix(query_tf, idf)

    save_matrix(query_tfidf, QUERY_TFIDF_FILE)

    print("\nMatrices guardadas en:")
    print(OUTPUT_DIR)

    # --------------------------
    # VALIDACIÓN PARA REPORTE
    # --------------------------
    print("\n" + "=" * 60)
    print("VALIDACIÓN PARA REPORTE")
    print("=" * 60)

    palabra = vocab[137]  # primera palabra del vocabulario (orden alfabético)
    indice  = 0         # su posición en el vocabulario

    print(f"\nPalabra elegida : '{palabra}' (índice {indice})")

    # Buscar un documento donde aparezca más de una vez
    for i, tokens in enumerate(documents):
        freq = tokens.count(palabra)
        if freq > 1:
            print(f"Documento       : #{i + 1}")
            print(f"Frecuencia real : {freq} veces en el documento preprocesado")
            print(f"Valor en TF     : matriz TF [renglón {i + 1}, columna {indice}] = {doc_tf[i][indice]}")
            break
    else:
        print("No se encontró ningún documento donde la palabra aparezca más de una vez.")
        print("Considera elegir otra palabra del vocabulario para el reporte.")

    print("\nPROCESO TERMINADO")


main()

if __name__ == "__main__":
    pass