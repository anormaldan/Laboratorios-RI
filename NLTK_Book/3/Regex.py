# Expresiones regulares completas
import nltk, re

# Corpus de palabras
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
print(f"Total de palabras en minúsculas: {len(wordlist)}")

# 1. Palabras terminadas en 'ed'
print("\n1. Palabras terminadas en 'ed' (primeras 10):")
ed_words = [w for w in wordlist if re.search('ed$', w)]
print(ed_words[:10])
print(f"Total: {len(ed_words)} palabras")

# 2. Patrón específico: ..j..t..
print("\n2. Patrón '^..j..t..$' (8 letras, j en 3ra, t en 6ta):")
pattern_words = [w for w in wordlist if re.search('^..j..t..$', w)]
print(pattern_words[:10])
print(f"Total: {len(pattern_words)} palabras")

# 3. Email opcional
print("\n3. Buscar 'email' o 'e-mail':")
test_words = ['email', 'e-mail', 'mail', 'e mail', 'e_mail']
for w in test_words:
    if re.search('^e-?mail$', w):
        print(f"  Encontrado: {w}")

# 4. T9 system - palabras que se escriben con 4653
print("\n4. Palabras T9 para teclas 4653 (g/h/i, m/n/o, j/k/l, d/e/f):")
t9_words = [w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]
print(t9_words)
print(f"Total: {len(t9_words)} palabras")

# 5. Chat words
print("\n5. Palabras de chat con patrón 'm+i+n+e+':")
chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
mine_words = [w for w in chat_words if re.search('^m+i+n+e+$', w)]
print(mine_words[:5])  # Mostrar solo 5

# 6. WSJ ejemplos
wsj = sorted(set(nltk.corpus.treebank.words()))
print("\n6. Ejemplos del Wall Street Journal:")
print("  Números decimales (primeros 5):")
decimal_words = [w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]
print(f"    {decimal_words[:5]}")

print("  Monedas (ej: US$):")
currency_words = [w for w in wsj if re.search('^[A-Z]+\$$', w)]
print(f"    {currency_words}")

print("  Años 4 dígitos (primeros 5):")
year_words = [w for w in wsj if re.search('^[0-9]{4}$', w)]
print(f"    {year_words[:5]}")