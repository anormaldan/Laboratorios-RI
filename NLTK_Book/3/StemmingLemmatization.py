# Stemming y lematización
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk import word_tokenize

print("=" * 60)
print("STEMMING Y LEMATIZACIÓN")
print("=" * 60)

raw = "DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government. Supreme executive power derives from a mandate from the masses."
tokens = word_tokenize(raw)

print(f"\nTexto original ({len(tokens)} tokens):")
print(" ".join(tokens[:15]), "...")

porter = PorterStemmer()
lancaster = LancasterStemmer()
wnl = WordNetLemmatizer()

print("\n" + "-" * 80)
print(f"{'Token':<15} {'Porter':<15} {'Lancaster':<15} {'WordNet':<15}")
print("-" * 80)

for token in tokens[:20]:  # Mostrar solo primeros 20
    print(f"{token:<15} {porter.stem(token):<15} {lancaster.stem(token):<15} {wnl.lemmatize(token):<15}")

print("\n" + "=" * 60)
print("EJEMPLO CON CLASE IndexedText (del libro)")
print("=" * 60)

class IndexedText:
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i) 
                                 for (i, word) in enumerate(text))
    
    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)
        print(f"\nConcordancia para '{word}' (stem: '{key}'):")
        if key in self._index:
            for i in self._index[key][:5]:  # Mostrar solo 5 ocurrencias
                lcontext = ' '.join(self._text[i-wc:i])
                rcontext = ' '.join(self._text[i:i+wc])
                ldisplay = f'{lcontext[-width:]:>{width}}'
                rdisplay = f'{rcontext[:width]:<{width}}'
                print(ldisplay, rdisplay)
        else:
            print(f"  No se encontró '{word}'")
    
    def _stem(self, word):
        return self._stemmer.stem(word).lower()

# Ejemplo con el texto del Santo Grial
print("\nProbando con el texto del Santo Grial...")
porter_stemmer = PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter_stemmer, grail)
text.concordance('lie')