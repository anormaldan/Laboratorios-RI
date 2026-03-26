# String Operations: concatenación, repetición, slicing, etc.
# Muestra operaciones básicas con strings

monty = 'Monty Python'
circus = "Monty Python's Flying Circus"
circus2 = 'Monty Python\'s Flying Circus'

print("String 1:", monty)
print("String 2:", circus)
print("String 3:", circus2)

# Concatenación y multiplicación
print("\nConcatenación 'very' + 'very' + 'very':", 'very' + 'very' + 'very')
print("Multiplicación 'very' * 3:", 'very' * 3)

# Índices y slicing
print("\nÍndices:")
print("monty[0]:", monty[0])
print("monty[3]:", monty[3])
print("monty[-1]:", monty[-1])
print("monty[6:10]:", monty[6:10])
print("monty[-12:-7]:", monty[-12:-7])

# find() y rfind()
print("\nBúsqueda:")
print("monty.find('Python'):", monty.find('Python'))
print("monty.rfind('Python'):", monty.rfind('Python'))

# in operator
print("\nOperador 'in':")
print("'Python' in monty:", 'Python' in monty)
print("'Java' in monty:", 'Java' in monty)

# join() y split()
silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
print("\nJoin y Split:")
print("' '.join(silly):", ' '.join(silly))
print("';'.join(silly):", ';'.join(silly))
print("monty.split():", monty.split())

# Métodos de strings
print("\nMétodos de strings:")
print("monty.lower():", monty.lower())
print("monty.upper():", monty.upper())
print("'  hello  '.strip():", '  hello  '.strip())
print("monty.replace('Python', 'Java'):", monty.replace('Python', 'Java'))