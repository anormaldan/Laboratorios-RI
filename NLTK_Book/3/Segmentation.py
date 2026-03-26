# Segmentación de oraciones y palabras
import nltk
from random import randint

print("=" * 60)
print("SEGMENTACIÓN DE TEXTO")
print("=" * 60)

# 1. Segmentación de oraciones
print("\n1. SEGMENTACIÓN DE ORACIONES")
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = nltk.sent_tokenize(text)
print(f"   Total de oraciones encontradas: {len(sents)}")
print("\n   Primeras 3 oraciones:")
for i, sent in enumerate(sents[:3]):
    print(f"   {i+1}. {sent[:80]}...")

# Estadísticas del corpus Brown
print("\n2. ESTADÍSTICAS DEL BROWN CORPUS")
brown_words = len(nltk.corpus.brown.words())
brown_sents = len(nltk.corpus.brown.sents())
avg_words = brown_words / brown_sents
print(f"   Total de palabras: {brown_words:,}")
print(f"   Total de oraciones: {brown_sents:,}")
print(f"   Promedio palabras/oración: {avg_words:.2f}")

# 2. Segmentación de palabras (algoritmo del libro)
print("\n3. SEGMENTACIÓN DE PALABRAS (algoritmo Brent)")
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words

def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = sum(len(word) + 1 for word in set(words))
    return text_size + lexicon_size

# Ejemplo del libro
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100100010010000100100010"

print("\n   Texto de ejemplo:")
print(f"   '{text}'")
print(f"   Longitud: {len(text)} caracteres")

print("\n   Segmentación 1 (solo espacios principales):")
words1 = segment(text, seg1)
print(f"   Resultado: {words1}")
print(f"   Costo: {evaluate(text, seg1)}")

print("\n   Segmentación 2 (espacios entre palabras):")
words2 = segment(text, seg2)
print(f"   Resultado: {words2}")
print(f"   Costo: {evaluate(text, seg2)}")

print("\n   Comparación:")
print(f"   Segmentación 1 tiene {len(words1)} 'palabras'")
print(f"   Segmentación 2 tiene {len(words2)} palabras")
print(f"   Segmentación 2 es mejor (costo más bajo)")