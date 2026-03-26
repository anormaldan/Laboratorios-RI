# Ejercicios NLTK Book - Capítulo 2

## Ejercicios "Your turn"

### 1.  Make up a few sentences of your own, by typing a name, equals sign, and a list of words, like this: ex1 = ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail'].
Repeat some of the other Python operations we saw earlier in 1, e.g., sorted(ex1), len(set(ex1)), ex1.count('the').

```python
# Entrada
ex1 = ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail']
print(sorted(ex1))
print(len(set(ex1)))
print(ex1.count('the'))
```
```python
# Salida
['Grail', 'Holy', 'Monty', 'Python', 'and', 'the']
6
1
```
### 2. Take a few minutes to define a sentence of your own and modify individual words and groups of words (slices) using the same methods used earlier.

```python
# Entrada
sentence = ['The', 'cat', 'is', 'under', 'the', 'table']
print(sentence)
sentence[1] = 'dog'
print(sentence)
sentence[3:5] = ['on', 'top', 'of']
print(sentence)
sentence[3:6] = []
print(sentence)
```
```python
# Salida
['The', 'cat', 'is', 'under', 'the', 'table']
['The', 'dog', 'is', 'under', 'the', 'table']
['The', 'dog', 'is', 'on', 'top', 'of', 'table']
['The', 'dog', 'is', 'table']
```

## Ejercicios

### ☼ Review 2 on lists and strings.

Define a string and assign it to a variable, e.g., my_string = 'My String' (but put something more interesting in the string). Print the contents of this variable in two ways, first by simply typing the variable name and pressing enter, then by using the print statement.
```python
# Entrada
str = "Frase ejemplo 1"
print(str)
```
```python
# Salida
Frase ejemplo 1
```

Try adding the string to itself using my_string + my_string, or multiplying it by a number, e.g., my_string * 3. Notice that the strings are joined together without any spaces. How could you fix this?
```python
# Entrada
str + " " + str
```
```python
# Salida
'Frase ejemplo 1 Frase ejemplo 1'
```
### ☼ Define a variable my_sent to be a list of words, using the syntax my_sent = ["My", "sent"] (but with your own words, or a favorite saying).

Use ' '.join(my_sent) to convert this into a string.
```python
# Entrada
my_sent = ["Esta", "es", "una", "frase", "de", "ejemplo"]
' '.join(my_sent)
```
```python
# Salida
'Esta es una frase de ejemplo'
```

Use split() to split the string back into the list form you had to start with.
```python
# Entrada
my_string = 'Esta es una frase de ejemplo'
my_string.split()
```
```python
# Salida
['Esta', 'es', 'una', 'frase', 'de', 'ejemplo']
```

### ☼ Define several variables containing lists of words, e.g., phrase1, phrase2, and so on. 
Join them together in various combinations (using the plus operator) to form whole sentences. What is the relationship between len(phrase1 + phrase2) and len(phrase1) + len(phrase2)?
```python
# Entrada
phrase1 = ['La', 'vida', 'es', 'bella']
phrase2 = ['Disfruta', 'cada', 'momento']
combined = phrase1 + phrase2
len_combined = len(combined)
len_individual = len(phrase1) + len(phrase2)
len_combined, len_individual
```
```python
# Salida
(7, 7)
```


