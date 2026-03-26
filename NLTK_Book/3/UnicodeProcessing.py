# Procesamiento de texto con Unicode
# Muestra cómo manejar archivos en codificaciones distintas a UTF-8

import nltk
import unicodedata

# Archivo en Latin-2
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
print("Ruta del archivo:", path)

# Leer mostrando contenido
print("\nContenido del archivo (Latin-2):")
with open(path, encoding='latin2') as f:
    for i, line in enumerate(f):
        if i < 3:  # Mostrar solo 3 líneas
            line = line.strip()
            print(f"Línea {i+1}: {line}")

# Mostrar codepoints
print("\n" + "="*50)
print("Representación con escape Unicode:")
with open(path, encoding='latin2') as f:
    for i, line in enumerate(f):
        if i < 2:
            line = line.strip()
            print(f"Línea {i+1}: {line.encode('unicode_escape')}")

# Caracteres especiales
print("\n" + "="*50)
print("Caracteres Unicode especiales:")
nacute = '\u0144'
print("nacute = '\\u0144':", nacute)
print("ord('ń'):", ord('ń'))
print("hex(324):", hex(324))

# Usando unicodedata
print("\n" + "="*50)
print("Información de caracteres con unicodedata:")
with open(path, encoding='latin2') as f:
    lines = f.readlines()
    line = lines[2]
    print("Línea 3:", line.strip())
    print("\nCaracteres no ASCII:")
    for c in line:
        if ord(c) > 127:
            print(f"  {c.encode('utf8')} U+{ord(c):04x} {unicodedata.name(c)}")