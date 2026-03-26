# Ejercicios de patrones regex
import re

print("=" * 60)
print("YOUR TURN: DESCRIBIR PATRONES REGEX")
print("=" * 60)

# Lista de patrones a describir
patterns = [
    (r'[a-zA-Z]+', "a. Letras mayúsculas y minúsculas"),
    (r'[A-Z][a-z]*', "b. Mayúscula seguida de minúsculas opcionales"),
    (r'p[aeiou]{,2}t', "c. 'p', luego 0-2 vocales, luego 't'"),
    (r'\d+(\.\d+)?', "d. Número entero con parte decimal opcional"),
    (r'[^aeiouAEIOU][aeiouAEIOU][^aeiouAEIOU]*', "e. No vocal, vocal, luego no vocales"),
    (r'\w+|[^\w\s]+', "f. Palabra O caracteres no-alfanuméricos (no espacios)"),
]

test_strings = [
    'Python', 'Hello', 'pat', 'pet', 'pyt', 'paeiot', '3', '3.14', '3.14159',
    'bat', 'cat', 'dog', 'hello!', 'world?', 'test123', '123test', 'a', 'A',
    'bc', 'Abc', 'ABc', 'pat', 'pet', 'pit', 'pot', 'put', 'pt', 'paet', 'paeiot'
]

print("\nTest strings disponibles:")
print(", ".join(test_strings))

print("\n" + "=" * 60)
print("RESULTADOS DE LOS PATRONES")
print("=" * 60)

for pattern, description in patterns:
    print(f"\n{description}")
    print(f"Patrón: {pattern}")
    print("Coincidencias:")
    matches = []
    for s in test_strings:
        if re.fullmatch(pattern, s):
            matches.append(s)
    
    if matches:
        print(f"  {', '.join(matches)}")
        print(f"  Total: {len(matches)} coincidencias")
    else:
        print("  Ninguna coincidencia")
    
    # Explicación
    if 'a.' in description:
        print("  Explicación: Cualquier secuencia de letras (1 o más)")
    elif 'b.' in description:
        print("  Explicación: Empieza con mayúscula, puede tener minúsculas después")
    elif 'c.' in description:
        print("  Explicación: p seguido de hasta 2 vocales, luego t")
    elif 'd.' in description:
        print("  Explicación: Número con o sin decimales")
    elif 'e.' in description:
        print("  Explicación: No vocal + vocal + cualquier cosa que no sea vocal")
    elif 'f.' in description:
        print("  Explicación: Palabras O puntuación/símbolos")

print("\n" + "=" * 60)
print("USANDO nltk.re_show()")
print("=" * 60)

# Ejemplo con nltk.re_show
test_text = "Hello world! Python 3.14 is great. Testing pat, pet, and pyt."
print(f"\nTexto de prueba: {test_text}")

for pattern, description in patterns[:3]:  # Solo primeros 3 para no saturar
    print(f"\n{description}")
    print(f"Patrón: {pattern}")
    # nltk.re_show muestra dónde coincide el patrón
    try:
        nltk.re_show(pattern, test_text)
    except:
        print("  (nltk.re_show no disponible en este entorno)")
        # Alternativa manual
        matches = re.findall(pattern, test_text)
        print(f"  Coincidencias encontradas: {matches}")