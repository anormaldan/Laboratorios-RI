# Buscar patrones "as X as Y"
import nltk

print("=" * 60)
print("YOUR TURN: BUSCAR PATRONES 'as X as Y'")
print("=" * 60)

print("\n1. EN EL CORPUS BROWN (categorías hobbies y learned):")
from nltk.corpus import brown

# Combinar hobbies y learned
hobbies_words = brown.words(categories=['hobbies'])
learned_words = brown.words(categories=['learned'])
combined_words = list(hobbies_words) + list(learned_words)
hobbies_learned = nltk.Text(combined_words)

print(f"   Total de palabras: {len(combined_words):,}")

# Buscar patrones "as ADJ as NOUN"
print("\n   Buscando patrones 'as <adjetivo> as <sustantivo>':")
found = hobbies_learned.findall(r"<as> <\w*> <as> <\w*>")
print(f"   Encontrados {len(found)} patrones")

if found:
    print("\n   Primeros 20 resultados:")
    for i, pattern in enumerate(found[:20]):
        print(f"   {i+1:2}. {pattern}")
    
    print("\n   Análisis de los patrones:")
    print("   - 'as ... as ...' es una construcción comparativa")
    print("   - El primer <\\w*> suele ser un adjetivo")
    print("   - El segundo <\\w*> suele ser un sustantivo")
    print("   - Ejemplos típicos: 'as good as gold', 'as white as snow'")

print("\n" + "=" * 60)
print("2. EN EL CORPUS GUTENBERG (libros completos):")
from nltk.corpus import gutenberg

# Tomar varios libros
books = ['austen-emma.txt', 'austen-persuasion.txt', 'bible-kjv.txt']
all_gutenberg = []

for book in books:
    words = gutenberg.words(book)
    all_gutenberg.extend(words)

gutenberg_text = nltk.Text(all_gutenberg)
print(f"   Total de palabras: {len(all_gutenberg):,}")

# Buscar patrones más específicos
print("\n   Buscando 'as X as Y' donde X tiene al menos 3 letras:")
found_gutenberg = gutenberg_text.findall(r"<as> <\w{3,}> <as> <\w*>")
print(f"   Encontrados {len(found_gutenberg)} patrones")

if found_gutenberg:
    print("\n   Primeros 15 resultados:")
    unique_patterns = sorted(set(found_gutenberg))
    for i, pattern in enumerate(unique_patterns[:15]):
        print(f"   {i+1:2}. {pattern}")

print("\n" + "=" * 60)
print("3. HIPÉRÓNIMOS CON 'X and other Ys':")
print("\n   Este patrón revela relaciones de hiperonimia:")
print("   'X and other Ys' sugiere que X es un tipo de Y")

# Ejemplo del libro
brown_text = nltk.Text(brown.words())
hypernyms = brown_text.findall(r"<\w*> <and> <other> <\w*>")

print(f"\n   Encontrados {len(hypernyms)} patrones de hiperonimia")
if hypernyms:
    print("\n   Ejemplos encontrados:")
    for i, h in enumerate(hypernyms[:15]):
        parts = h.split()
        x = parts[0]
        y = parts[3]
        print(f"   {i+1:2}. '{x}' es un tipo de '{y}'")