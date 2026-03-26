# Aplicaciones de regex
import nltk, re

print("=" * 60)
print("APLICACIONES DE EXPRESIONES REGULARES")
print("=" * 60)

# 1. Extraer vocales
word = 'supercalifragilisticexpialidocious'
vocales = re.findall(r'[aeiou]', word)
print(f"\n1. Extraer vocales de '{word}':")
print(f"   Vocales encontradas: {vocales}")
print(f"   Número de vocales: {len(vocales)}")
print(f"   Vocales únicas: {sorted(set(vocales))}")

# 2. Frecuencia de secuencias de vocales
print("\n2. Frecuencia de secuencias de dos o más vocales en WSJ:")
wsj = set(nltk.corpus.treebank.words())
fd = nltk.FreqDist(vs for w in wsj for vs in re.findall(r'[aeiou]{2,}', w))
print("   Secuencias más comunes:")
for seq, count in fd.most_common(10):
    print(f"     {seq}: {count}")

# 3. Comprimir texto (eliminar vocales internas)
def compress(word):
    regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
    pieces = re.findall(regexp, word)
    return ''.join(pieces)

print("\n3. Comprimir texto (eliminar vocales internas):")
english_udhr = nltk.corpus.udhr.words('English-Latin1')
print("   Primeras 20 palabras del UDHR (Inglés):")
print("   Original:", ' '.join(english_udhr[:20]))
print("   Comprimido:", ' '.join(compress(w) for w in english_udhr[:20]))

# 4. Stemming con regex
def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    result = re.findall(regexp, word)
    if result:
        stem_part, suffix = result[0]
        return stem_part
    return word

print("\n4. Stemming con expresiones regulares:")
raw = "DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government."
tokens = nltk.word_tokenize(raw)
print("   Original:", tokens[:10])
print("   Stemmed:", [stem(t) for t in tokens[:10]])

# 5. Búsqueda en texto tokenizado
print("\n5. Búsqueda de patrones en Moby Dick:")
moby = nltk.Text(nltk.corpus.gutenberg.words('melville-moby_dick.txt'))
print("   Patrón: 'a <cualquier palabra> man'")
matches = moby.findall(r"<a> (<.*>) <man>")
print(f"   Encontrados {len(matches)} resultados")
print("   Primeros 10:", matches[:10])