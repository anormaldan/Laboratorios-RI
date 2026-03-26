"""
Laboratorio 6: Recuperación de Información
Autores: Nava Campos Alejandro Dante, Ortega Pérez Andrea, y Tiro Cuanenemi Jonathan.

Objetivo:
Obtener la similitud entre consultas y documentos empleando coseno.

Entradas (matrices generadas en Lab 5):
  Lab5/output/matrices/documentos_tfidf.txt
  Lab5/output/matrices/consultas_tfidf.txt
  Lab5/output/matrices/documentos_tf.txt
  Lab5/output/matrices/consultas_tf.txt

Salidas:
  output/CACM_tf_idf_rels.txt
  output/CACM_tf_rels.txt
"""

import os
import math
import numpy as np

# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LAB5_DIR   = os.path.join(BASE_DIR, 'Lab5', 'output', 'matrices')

DOC_TFIDF_FILE   = os.path.join(LAB5_DIR, 'documentos_tfidf.txt')
QUERY_TFIDF_FILE = os.path.join(LAB5_DIR, 'consultas_tfidf.txt')
DOC_TF_FILE      = os.path.join(LAB5_DIR, 'documentos_tf.txt')
QUERY_TF_FILE    = os.path.join(LAB5_DIR, 'consultas_tf.txt')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


# ---------------------------------------------------------
# 1. Leer matriz guardada (ya procesada en Lab 5)
# ---------------------------------------------------------

def leer_matriz(ruta):
    """Lee una matriz numérica desde archivo, una fila por línea."""
    matriz = []
    with open(ruta, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                fila = [float(x) for x in line.split()]
                matriz.append(fila)
    return np.array(matriz)


# ---------------------------------------------------------
# 2. Similitud coseno
# ---------------------------------------------------------

def coseno(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------
# 3. Recuperación: coseno entre cada query y todos los docs
# ---------------------------------------------------------

def generar_resultados(matriz_docs, matriz_queries, nombre_archivo):
    """
    Para cada query calcula el coseno con todos los documentos,
    descarta similitud == 0, ordena descendente y guarda.
    Formato: QID DocID Similitud
    """
    os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)

    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        for i, query in enumerate(matriz_queries):
            sims = []

            for j, doc in enumerate(matriz_docs):
                sim = coseno(query, doc)
                if sim > 0:
                    sims.append((j + 1, sim))

            # Ordenar descendente por similitud
            sims.sort(key=lambda x: x[1], reverse=True)

            for doc_id, sim in sims:
                f.write(f"{i+1:03d} {doc_id} {sim:.4f}\n")

    print(f"  Guardado: {nombre_archivo}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("LABORATORIO 6: SIMILITUD COSENO - MODELO VECTORIAL")
    print("=" * 60)

    # --------------------------------------------------
    # Cargar matrices generadas en Lab 5
    # --------------------------------------------------
    print("\n[1] Cargando matrices del Lab 5...")

    for ruta in [DOC_TFIDF_FILE, QUERY_TFIDF_FILE, DOC_TF_FILE, QUERY_TF_FILE]:
        if not os.path.exists(ruta):
            print(f"  ERROR: No se encontró '{ruta}'")
            print("  Asegúrate de haber ejecutado Lab5.py primero.")
            return

    doc_tfidf   = leer_matriz(DOC_TFIDF_FILE)
    query_tfidf = leer_matriz(QUERY_TFIDF_FILE)
    doc_tf      = leer_matriz(DOC_TF_FILE)
    query_tf    = leer_matriz(QUERY_TF_FILE)

    print(f"  Documentos  (TF-IDF): {doc_tfidf.shape}")
    print(f"  Queries     (TF-IDF): {query_tfidf.shape}")
    print(f"  Documentos  (TF)    : {doc_tf.shape}")
    print(f"  Queries     (TF)    : {query_tf.shape}")

    # --------------------------------------------------
    # Generar resultados TF-IDF
    # --------------------------------------------------
    print("\n[2] Calculando similitudes TF-IDF...")
    tfidf_out = os.path.join(OUTPUT_DIR, 'CACM_tf_idf_rels.txt')
    generar_resultados(doc_tfidf, query_tfidf, tfidf_out)

    # --------------------------------------------------
    # Generar resultados TF
    # --------------------------------------------------
    print("\n[3] Calculando similitudes TF...")
    tf_out = os.path.join(OUTPUT_DIR, 'CACM_tf_rels.txt')
    generar_resultados(doc_tf, query_tf, tf_out)

    print("\nPROCESO TERMINADO")
    print(f"Archivos generados en: {OUTPUT_DIR}")


main()

if __name__ == "__main__":
    pass