# Formateo de salida
import nltk
from textwrap import fill

print("=" * 60)
print("FORMATEO DE SALIDA")
print("=" * 60)

# 1. join() básico
print("\n1. MÉTODO join()")
silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
print("   Lista original:", silly)
print("   ' '.join(silly):", ' '.join(silly))
print("   ';'.join(silly):", ';'.join(silly))
print("   '-'.join(silly):", '-'.join(silly))

# 2. format() básico
print("\n2. MÉTODO format()")
print("   '{} wants a {} {}'.format('Lee', 'sandwich', 'for lunch'):")
print("   Resultado:", '{} wants a {} {}'.format('Lee', 'sandwich', 'for lunch'))

print("\n   'from {} to {}'.format('A', 'B'):", 'from {} to {}'.format('A', 'B'))
print("   'from {1} to {0}'.format('A', 'B'):", 'from {1} to {0}'.format('A', 'B'))

# 3. Formateo con ancho
print("\n3. FORMATEO CON ANCHO")
print("   '{:6}'.format(123):", '{:6}'.format(123))
print("   '{:6}'.format('abc'):", '{:6}'.format('abc'))
print("   '{:<6}'.format(123):", '{:<6}'.format(123))
print("   '{:>6}'.format('abc'):", '{:>6}'.format('abc'))

# 4. Tabular datos
print("\n4. TABULACIÓN DE DATOS")
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in ['news', 'romance', 'hobbies']
    for word in nltk.corpus.brown.words(categories=genre)
    if word in ['can', 'could', 'may', 'might', 'must', 'will']
)

genres = ['news', 'romance', 'hobbies']
modals = ['can', 'could', 'may', 'might', 'must', 'will']

print("\n   Frecuencia de modales por género:")
print(' ' * 12 + ' '.join(f'{m:>6}' for m in modals))
print('   ' + '-' * 50)
for g in genres:
    print(f'   {g:10}', end=' ')
    for m in modals:
        print(f'{cfd[g][m]:6}', end=' ')
    print()

# 5. Escritura en archivo
print("\n5. ESCRITURA EN ARCHIVO")
with open('output.txt', 'w') as f:
    for word in silly:
        f.write(word + '\n')
print("   Archivo 'output.txt' creado con éxito.")

# 6. Text wrapping
print("\n6. TEXT WRAPPING")
saying = ['After', 'all', 'is', 'said', 'and', 'done', ',', 
          'more', 'is', 'said', 'than', 'done', '.']
print("   Lista original:", saying)
pieces = [f"{word} ({len(word)})" for word in saying]
output = ' '.join(pieces)
print("   Con longitud:", output)
wrapped = fill(output, width=40)
print("\n   Con textwrap.fill() (ancho 40):")
print(wrapped)

# 7. Formateo con variables
print("\n7. FORMATEO CON VARIABLES EN ANCHO")
width = 15
print(f"   Ancho variable: {width}")
print(f"   '{{:{{width}}}}'.format('Monty Python', width={width}):")
print(f"   Resultado: {{:{width}}}".format('Monty Python'))