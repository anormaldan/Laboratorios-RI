# Tokenización completa
import nltk, re

print("=" * 60)
print("MÉTODOS DE TOKENIZACIÓN")
print("=" * 60)

raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone though), 'I won't have any pepper in my kitchen AT ALL. Soup does very well without--Maybe it's always pepper that makes people hot-tempered,'..."""

print(f"\nTexto original (primeros 150 caracteres):")
print(raw[:150], "...")

# Método 1: split() simple
print("\n" + "-" * 50)
print("1. Método split() simple (primeros 10 tokens):")
tokens_split = raw.split()
print("   Resultado:", tokens_split[:10])
print("   Problema: 'Duchess,' mantiene la coma")

# Método 2: re.split con espacios
print("\n2. Método re.split(r'[ \\t\\n]+'):")
tokens_resplit = re.split(r'[ \t\n]+', raw)
print("   Resultado:", tokens_resplit[:10])
print("   Mejor, pero aún tiene problemas con puntuación")

# Método 3: re.findall con \w+
print("\n3. Método re.findall(r'\\w+'):")
tokens_words = re.findall(r'\w+', raw)
print("   Resultado:", tokens_words[:10])
print("   Pierde: I'M se convierte en I, M")

# Método 4: re.findall más complejo
print("\n4. Método re.findall con apóstrofes/hífenes:")
pattern_complex = r"\w+(?:[-']\w+)*"
tokens_complex = re.findall(pattern_complex, raw)
print("   Resultado:", tokens_complex[:10])
print("   Mejor: maneja I'M, won't, hot-tempered")

# Método 5: regexp_tokenize de NLTK
print("\n5. Método nltk.regexp_tokenize:")
pattern = r'''(?x)          # verbose flag
    (?:[A-Z]\.)+           # abreviaciones
    | \w+(?:[-']\w+)*      # palabras con hífen/apóstrofe
    | \$?\d+(?:\.\d+)?%?   # dinero y porcentajes
    | \.\.\.               # elipsis
    | [][.,;'?():-_"]      # signos de puntuación
'''
tokens_nltk = nltk.regexp_tokenize(raw, pattern)
print("   Resultado:", tokens_nltk[:15])
print("   Mejor aún: separa puntuación correctamente")

print("\n" + "=" * 60)
print("COMPARACIÓN DE RESULTADOS")
print("=" * 60)

methods = [
    ("split()", len(tokens_split)),
    ("re.split", len(tokens_resplit)),
    ("re.findall \\w+", len(tokens_words)),
    ("re.findall complejo", len(tokens_complex)),
    ("nltk.regexp_tokenize", len(tokens_nltk))
]

print(f"\nNúmero de tokens por método:")
for method, count in methods:
    print(f"  {method:<25} {count:>4} tokens")