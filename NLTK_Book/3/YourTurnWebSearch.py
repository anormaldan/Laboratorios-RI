# Ejercicio: Buscar "the of" en web
print("=" * 60)
print("YOUR TURN: ¿Es 'the of' una colocación frecuente?")
print("=" * 60)

print("\nAnálisis:")
print("-" * 50)
print("""
Al buscar "the of" entre comillas en un motor de búsqueda,
encontramos millones de resultados. Sin embargo, esto NO significa
que 'the of' sea una colocación gramatical válida en inglés.

La razón es que los motores de búsqueda encuentran estas dos palabras
cerca una de la otra en contextos como:

1. "one of the most"       -> ... of the ...
2. "some of the best"      -> ... of the ...
3. "many of the people"    -> ... of the ...
4. "out of the question"   -> ... of the ...

En todos estos casos, 'the' y 'of' están separados por otras palabras
o pertenecen a construcciones gramaticales más largas.

Conclusión:
El alto conteo se debe a que 'the' y 'of' son palabras muy frecuentes
en inglés que a menudo aparecen cerca en estructuras gramaticales,
pero 'the of' por sí solo NO es una colocación válida.
""")

print("\nEjemplos reales (del corpus Brown):")
print("-" * 50)
from nltk.corpus import brown
brown_text = nltk.Text(brown.words())

# Buscar ejemplos donde 'the' y 'of' estén cerca
import re
sample = ' '.join(brown.words()[:1000])
matches = re.findall(r'\bof\s+the\b', sample)
print(f"En 1000 palabras del Brown Corpus, 'of the' aparece {len(matches)} veces")
if matches:
    print("Ejemplos encontrados:", matches[:5])