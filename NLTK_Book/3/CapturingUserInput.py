# Capturing User Input
# Permite capturar texto ingresado por el usuario y procesarlo con NLTK

from nltk import word_tokenize

s = input("Enter some text: ")
print("You typed", len(word_tokenize(s)), "words.")
print("Tokens:", word_tokenize(s))