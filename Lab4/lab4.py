"""
Laboratorio 4: Recuperación de Información
Autores: Nava Campos Alejandro Dante, Ortega Pérez Andrea, y Tiro Cuanenemi Jonathan.

Objetivo: Obtener el vocabulario de la colección CACM a partir de los documentos
preprocesados generados en el Laboratorio 3. Se aplica un mecanismo de reducción
de vocabulario basado en Document Frequency (DF) con umbral piso y techo:

  - DF mínimo (piso) : se eliminan términos que aparecen en menos de MIN_DF
                       documentos distintos. Son demasiado raros para conectar
                       documentos con consultas.
  - DF máximo (techo): los términos que aparecen en más del MAX_DF_RATIO de los
                       documentos NO se descartan, sino que se guardan en un archivo
                       separado (vocabularioAltoDF.txt). Aunque son omnipresentes en
                       la colección y no discriminan entre documentos de forma
                       individual, pueden representar el contexto temático central
                       de la colección y ser útiles en modelos como BM25 o en
                       análisis de relevancia posteriores.

Este enfoque es la base del esquema de pesado TF-IDF ampliamente utilizado en IR.

Entrada esperada: output/docs/documentos_snowball.txt  (generado por el Lab 3)
Salidas:
  output/vocabulary/vocabulario.txt        — vocabulario completo ordenado alfabéticamente
  output/vocabulary/vocabularioReducido.txt — términos con DF en rango [MIN_DF, MAX_DF]
  output/vocabulary/vocabularioAltoDF.txt  — términos eliminados por techo (alto DF)
"""

import os
from collections import Counter


# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_FILE = os.path.join(BASE_DIR, 'Lab3', 'output', 'docs', 'documentos_snowball.txt')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'vocabulary')
VOCAB_FILE = os.path.join(OUTPUT_DIR, 'vocabulario.txt')
VOCAB_RED_FILE = os.path.join(OUTPUT_DIR, 'vocabularioReducido.txt')
VOCAB_ALTO_DF_FILE = os.path.join(OUTPUT_DIR, 'vocabularioAltoDF.txt')

# Parámetros de reducción por Document Frequency
MIN_DF       = 2      # Piso : término debe aparecer en al menos 2 documentos distintos
MAX_DF_RATIO = 0.80   # Techo: término no debe aparecer en más del 80% de los documentos


def load_documents(filepath: str):
    """
    Lee el archivo preprocesado (formato: num | titulo | texto) y retorna
    dos estructuras:
      - term_freq  : frecuencia total de cada término en toda la colección (TF global).
      - doc_freq   : número de documentos distintos en que aparece cada término (DF).

    NOTA: El campo 'texto' (abstract) puede estar vacío en algunos documentos.
    El formato real del archivo es:
        "num | titulo |"           ← documento sin abstract (solo 2 separadores)
        "num | titulo | abstract"  ← documento con abstract

    Ambos casos son válidos y se cuentan como documentos. La corrección clave
    es aceptar líneas con 2 o 3 partes al hacer split(' | ', 2), en lugar de
    descartar las que no producen exactamente 3 partes.

    Parámetros
    ----------
    filepath : str
        Ruta al archivo de documentos preprocesados.

    Retorna
    -------
    tuple (Counter, Counter, int)
        term_freq  — frecuencia total por término.
        doc_freq   — document frequency por término.
        total_docs — número total de documentos leídos.
    """
    term_freq = Counter()
    doc_freq = Counter()
    total_docs = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Saltar líneas vacías

            parts = line.split(' | ', 2)

            # CORRECCIÓN: se aceptan tanto líneas con abstract (3 partes)
            # como sin abstract (2 partes). Antes, las líneas de 2 partes
            # eran descartadas por el 'if len(parts) != 3', lo que provocaba
            # que ~1,621 documentos no se contabilizaran en total_docs ni
            # contribuyeran al vocabulario ni al cálculo de DF.
            if len(parts) < 2:
                continue  # Saltar líneas realmente malformadas (sin separador)

            titulo = parts[1]
            texto  = parts[2] if len(parts) == 3 else ''  # abstract puede ser vacío
            total_docs += 1

            # Concatenar título y texto (el texto puede ser cadena vacía)
            texto_completo = f"{titulo} {texto}"

            # Tokenizar y eliminar números y tokens de un solo carácter
            tokens = [t for t in texto_completo.split() if t.isalpha() and len(t) > 1]
            if not tokens:
                continue

            # Actualizar frecuencia total
            term_freq.update(tokens)

            # Actualizar document frequency (cada término se cuenta una sola vez por doc)
            doc_freq.update(set(tokens))

    return term_freq, doc_freq, total_docs


def build_vocabulary(term_freq: Counter) -> list:
    """
    Construye el vocabulario completo: conjunto de términos únicos
    ordenados alfabéticamente.

    Parámetros
    ----------
    term_freq : Counter
        Frecuencia total de cada término en la colección.

    Retorna
    -------
    list
        Lista de términos únicos ordenados alfabéticamente.
    """
    vocab = sorted(term for term in term_freq if term.strip())
    return vocab


def reduce_vocabulary(doc_freq: Counter, total_docs: int,
                      min_df: int, max_df_ratio: float):
    """
    Clasifica el vocabulario en tres grupos según Document Frequency (DF):

      1. vocabularioReducido : términos con DF en rango [min_df, max_df].
                               Son los más informativos para recuperación.

      2. vocabularioAltoDF   : términos con DF > max_df (eliminados por techo).
                               No se descartan permanentemente: al ser omnipresentes
                               pueden representar el núcleo temático de la colección
                               y ser útiles en modelos de pesado como BM25.

      3. Eliminados por piso : términos con DF < min_df. Son demasiado raros
                               para conectar documentos con consultas de forma
                               confiable, por lo que sí se descartan.

    Parámetros
    ----------
    doc_freq     : Counter — document frequency de cada término.
    total_docs   : int     — número total de documentos en la colección.
    min_df       : int     — umbral mínimo de DF (piso).
    max_df_ratio : float   — proporción máxima de documentos permitida (techo).

    Retorna
    -------
    tuple (list, list)
        vocab_reducido — términos en rango normal, ordenados alfabéticamente.
        vocab_alto_df  — términos de alto DF, ordenados alfabéticamente.
    """
    max_df = int(max_df_ratio * total_docs)

    vocab_reducido = sorted(
        term for term, df in doc_freq.items()
        if min_df <= df <= max_df and term.strip()
    )

    vocab_alto_df = sorted(
        term for term, df in doc_freq.items()
        if df > max_df and term.strip()
    )

    return vocab_reducido, vocab_alto_df


def save_vocabulary(vocab: list, filepath: str) -> None:
    """
    Guarda el vocabulario en un archivo de texto, un término por línea.

    Parámetros
    ----------
    vocab    : list — lista de términos ordenados.
    filepath : str  — ruta del archivo de salida.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))


def main():
    print("=" * 60)
    print("LABORATORIO 4: OBTENCIÓN DE VOCABULARIO - COLECCIÓN CACM")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PASO 1: Cargar documentos y calcular frecuencias
    # ------------------------------------------------------------------
    print(f"\n[1] Leyendo archivo: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"  ERROR: No se encontró el archivo '{INPUT_FILE}'.")
        print("  Asegúrate de haber ejecutado el Lab 3 primero (main.py).")
        return

    term_freq, doc_freq, total_docs = load_documents(INPUT_FILE)
    print(f"  Documentos leídos                 : {total_docs:,}")
    print(f"  Términos totales (con repetición) : {sum(term_freq.values()):,}")

    # ------------------------------------------------------------------
    # PASO 2: Construir vocabulario completo
    # ------------------------------------------------------------------
    print("\n[2] Construyendo vocabulario completo...")
    vocabulario = build_vocabulary(term_freq)
    print(f"  Longitud del vocabulario          : {len(vocabulario):,} términos únicos")

    # ------------------------------------------------------------------
    # PASO 3: Clasificar vocabulario por Document Frequency
    # ------------------------------------------------------------------
    max_df_abs = int(MAX_DF_RATIO * total_docs)
    print(f"\n[3] Clasificando vocabulario por Document Frequency...")
    print(f"  Piso  — DF mínimo : >= {MIN_DF} documentos")
    print(f"  Techo — DF máximo : <= {max_df_abs} documentos ({MAX_DF_RATIO*100:.0f}% de {total_docs})")

    vocabulario_reducido, vocab_alto_df = reduce_vocabulary(
        doc_freq, total_docs, MIN_DF, MAX_DF_RATIO
    )

    eliminados_piso  = sum(1 for df in doc_freq.values() if df < MIN_DF)
    eliminados_techo = len(vocab_alto_df)
    eliminados_total = len(vocabulario) - len(vocabulario_reducido)

    print(f"  Eliminados por piso  (DF < {MIN_DF})      : {eliminados_piso:,} términos (descartados)")
    print(f"  Separados por techo  (DF > {max_df_abs})  : {eliminados_techo:,} términos (guardados en vocabularioAltoDF.txt)")
    print(f"  Longitud del vocabulario reducido  : {len(vocabulario_reducido):,} términos únicos")

    # Mostrar los términos de alto DF para análisis
    if vocab_alto_df:
        print(f"\n  Términos de alto DF (contexto temático de la colección):")
        for term in vocab_alto_df:
            print(f"    '{term}'  →  DF = {doc_freq[term]:,} docs "
                  f"({doc_freq[term]/total_docs*100:.1f}%)")

    # ------------------------------------------------------------------
    # PASO 4: Guardar archivos de salida
    # ------------------------------------------------------------------
    print(f"\n[4] Guardando archivos de vocabulario...")
    save_vocabulary(vocabulario, VOCAB_FILE)
    print(f"  ✔  {VOCAB_FILE}")
    save_vocabulary(vocabulario_reducido, VOCAB_RED_FILE)
    print(f"  ✔  {VOCAB_RED_FILE}")
    save_vocabulary(vocab_alto_df, VOCAB_ALTO_DF_FILE)
    print(f"  ✔  {VOCAB_ALTO_DF_FILE}")

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Total de documentos              : {total_docs:,}")
    print(f"  Vocabulario original             : {len(vocabulario):,} términos")
    print(f"  Vocabulario reducido             : {len(vocabulario_reducido):,} términos")
    print(f"  Vocabulario alto DF              : {len(vocab_alto_df):,} términos")
    print(f"  Términos eliminados (piso)       : {eliminados_piso:,} "
          f"({eliminados_piso / len(vocabulario) * 100:.1f}%)")
    print(f"  Reducción efectiva del vocab     : {eliminados_total:,} "
          f"({eliminados_total / len(vocabulario) * 100:.1f}% del vocabulario original)")
    print("=" * 60)
    print("\nPROCESAMIENTO COMPLETADO")


main()
if __name__ == '__main__':
    pass