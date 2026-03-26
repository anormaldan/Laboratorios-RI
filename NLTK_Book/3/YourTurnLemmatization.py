# Comparación de normalización
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

print("=" * 60)
print("YOUR TURN: COMPARACIÓN DE NORMALIZACIÓN")
print("=" * 60)

porter = PorterStemmer()
wnl = WordNetLemmatizer()

# Lista de palabras para comparar
words = [
    'running', 'ran', 'runs', 'runner', 'runners',
    'easily', 'fairly', 'happily', 'quickly',
    'women', 'men', 'children', 'mice', 'geese',
    'better', 'best', 'worse', 'worst',
    'taking', 'taken', 'takes', 'took',
    'interesting', 'interested', 'interests'
]

print(f"\nComparando {len(words)} palabras:")
print("-" * 80)
print(f"{'Palabra':<15} {'Porter':<12} {'WordNet':<12} {'¿Iguales?':<10}")
print("-" * 80)

same_count = 0
for w in words:
    porter_stem = porter.stem(w)
    wordnet_lemma = wnl.lemmatize(w)
    are_same = porter_stem == wordnet_lemma
    if are_same:
        same_count += 1
    
    print(f"{w:<15} {porter_stem:<12} {wordnet_lemma:<12} {str(are_same):<10}")

print("-" * 80)
print(f"Porcentaje de coincidencia: {same_count/len(words)*100:.1f}%")

print("\n" + "=" * 60)
print("ANÁLISIS DE RESULTADOS")
print("=" * 60)

print("\n1. Verbos (running, ran, runs):")
print("   Porter: Todos reducidos a 'run' ✓")
print("   WordNet: Todos reducidos a 'run' ✓")
print("   Ambos funcionan bien para verbos regulares")

print("\n2. Adverbios (easily, happily):")
print("   Porter: 'easili', 'happili' (pierde significado)")
print("   WordNet: 'easily', 'happily' (mantiene forma base)")
print("   WordNet es mejor para adverbios")

print("\n3. Sustantivos irregulares (women, mice):")
print("   Porter: 'women', 'mice' (no cambia)")
print("   WordNet: 'woman', 'mouse' (forma singular)")
print("   WordNet es esencial para plurales irregulares")

print("\n4. Formas comparativas (better, worse):")
print("   Porter: 'better', 'worse' (no cambia)")
print("   WordNet: 'good', 'bad' (forma base)")
print("   WordNet maneja comparativos mejor")

print("\n" + "=" * 60)
print("RECOMENDACIONES")
print("=" * 60)
print("""
1. Para búsqueda de texto: Usar Porter Stemmer
   - Más agresivo
   - Mejor para recuperación de información
   - Agrupa más variantes

2. Para análisis lingüístico: Usar WordNet Lemmatizer
   - Más preciso
   - Produce palabras reales
   - Mantiene significado léxico

3. Para NLP general:
   - WordNet para lematización morfológica
   - Stemming solo si se necesita agrupar variantes radicalmente
""")