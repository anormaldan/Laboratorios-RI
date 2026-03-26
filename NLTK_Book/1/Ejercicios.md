# Ejercicios NLTK Book - Capítulo 1

## Ejemplos del libro NLTK Book
### 1.1 Getting started with Python
```python
# Entrada
 1 + 5 * 2 - 3
```

```python
# Salida
8
```

### 1.2 Getting Started with NLTK
```python
# Descarga del corpus NLTK Book
import nltk
nltk.download('book')
```

```python
# Importación de textos del NLTK Book
from nltk.book import *
```

```python
# Salida
*** Introductory Examples for the NLTK Book ***
Loading text1, ..., text9 and sent1, ..., sent9
Type the name of the text or sentence to view it.
Type: 'texts()' or 'sents()' to list the materials.
text1: Moby Dick by Herman Melville 1851
text2: Sense and Sensibility by Jane Austen 1811
text3: The Book of Genesis
text4: Inaugural Address Corpus
text5: Chat Corpus
text6: Monty Python and the Holy Grail
text7: Wall Street Journal
text8: Personals Corpus
text9: The Man Who Was Thursday by G . K . Chesterton 1908
```

```python
# Seleccion de corpus
text1
<Text: Moby Dick by Herman Melville 1851>
text2
<Text: Sense and Sensibility by Jane Austen 1811>
```

### 1.3 Searching text
```python
text1.concordance("monstrous")
```
```python
Displaying 11 of 11 matches:
ong the former , one was of a most monstrous size . ... This came towards us ,
ON OF THE PSALMS . " Touching that monstrous bulk of the whale or ork we have r
ll over with a heathenish array of monstrous clubs and spears . Some were thick
d as you gazed , and wondered what monstrous cannibal and savage could ever hav
that has survived the flood ; most monstrous and most mountainous ! That Himmal
they might scout at Moby Dick as a monstrous fable , or still worse and more de
th of Radney .'" CHAPTER 55 Of the monstrous Pictures of Whales . I shall ere l
ing Scenes . In connexion with the monstrous pictures of whales , I am strongly
ere to enter upon those still more monstrous stories of them which are to be fo
ght have been rummaged out of this monstrous cabinet there is no telling . But
of Whale - Bones ; for Whales of a monstrous size are oftentimes cast up dead u
```

```python
# Entradas
text1.similar("monstrous")
text2.similar("monstrous")
```
```python
# Salidas
mean part maddens doleful gamesome subtly uncommon careful untoward
exasperate loving passing mouldy christian few true mystifying
imperial modifies contemptible

very heartily so exceedingly remarkably as vast a great amazingly
extremely good sweet
```
```python
# Entrada
text2.common_contexts(["monstrous", "very"])
```
```python
# Salida
a_pretty is_pretty am_glad be_glad a_lucky
```

```python
# Entrada
text3.generate()
```
```python
# Salida
Building ngram index...
laid by her , and said unto Cain , Where art thou , and said , Go to ,
I will not do it for ten ' s sons ; we dreamed each man according to
their generatio the firstborn said unto Laban , Because I said , Nay ,
but Sarah shall her name be . , duke Elah , duke Shobal , and Akan .
and looked upon my affliction . Bashemath Ishmael ' s blood , but Isra
for as a prince hast thou found of all the cattle in the valley , and
the wo The
"laid by her , and said unto Cain , Where art thou , and said , Go to ,\nI will not do it for ten ' s sons ; we dreamed each man according to\ntheir generatio the firstborn said unto Laban , Because I said , Nay ,\nbut Sarah shall her name be . , duke Elah , duke Shobal , and Akan .\nand looked upon my affliction . Bashemath Ishmael ' s blood , but Isra\nfor as a prince hast thou found of all the cattle in the valley , and\nthe wo The"
```

### 1.4 Counting vocabulary
```python
# Entrada
len(text3)
```

```python
# Salida
44764
```
```python
# Entrada
sorted(set(text3))
```

```python
# Salida
['!', "'", '(', ')', ',', ',)', '.', '.)', ':', ';', ';)', '?', '?)', 'A', 'Abel', 'Abelmizraim', 'Abidah', 'Abide', 'Abimael', 'Abimelech', 'Abr', 'Abrah', 'Abraham', 'Abram', 'Accad', 'Achbor', 'Adah', 'Adam', 'Adbeel', 'Admah', 'Adullamite', 'After', 'Aholibamah', 'Ahuzzath', 'Ajah', 'Akan', 'All', 'Allonbachuth', 'Almighty', 'Almodad', 'Also', 'Alvah', 'Alvan', 'Am', 'Amal', 'Amalek', 'Amalekites', 'Ammon', 'Amorite', 'Amorites', 'Amraphel', 'An', 'Anah', 'Anamim', 'And', 'Aner', 'Angel', 'Appoint', 'Aram', 'Aran', 'Ararat', 'Arbah', 'Ard', 'Are', 'Areli', 'Arioch', 'Arise', 'Arkite', 'Arodi', 'Arphaxad', 'Art', 'Arvadite', 'As', 'Asenath', 'Ashbel', 'Asher', 'Ashkenaz', 'Ashteroth', 'Ask', 'Asshur', 'Asshurim', 'Assyr', 'Assyria', 'At', 'Atad', 'Avith', 'Baalhanan', 'Babel', 'Bashemath', 'Be', 'Because', 'Becher', 'Bedad', 'Beeri', 'Beerlahairoi', 'Beersheba', 'Behold', 'Bela', 'Belah', 'Benam', 'Benjamin', 'Beno', 'Beor', 'Bera', 'Bered', 'Beriah', 'Bethel', 'Bethlehem', 'Bethuel', 'Beware', 'Bilhah', 'Bilhan', 'Binding', 'Birsha', 'Bless', 'Blessed', 'Both', 'Bow', 'Bozrah', 'Bring', 'But', 'Buz', 'By', 'Cain', 'Cainan', 'Calah', 'Calneh', 'Can', 'Cana', 'Canaan', 'Canaanite', 'Canaanites', 'Canaanitish', 'Caphtorim', 'Carmi', 'Casluhim', 'Cast', 'Cause', 'Chaldees', 'Chedorlaomer', 'Cheran', 'Cherubims', 'Chesed', 'Chezib', 'Come', 'Cursed', 'Cush', 'Damascus', 'Dan', 'Day', 'Deborah', 'Dedan', 'Deliver', 'Diklah', 'Din', 'Dinah', 'Dinhabah', 'Discern', 'Dishan', 'Dishon', 'Do', 'Dodanim', 'Dothan', 'Drink', 'Duke', 'Dumah', 'Earth', 'Ebal', 'Eber', 'Edar', 'Eden', 'Edom', 'Edomites', 'Egy', 'Egypt', 'Egyptia', 'Egyptian', 'Egyptians', 'Ehi', 'Elah', 'Elam', 'Elbethel', 'Eldaah', 'EleloheIsrael', 'Eliezer', 'Eliphaz', 'Elishah', 'Ellasar', 'Elon', 'Elparan', 'Emins', 'En', 'Enmishpat', 'Eno', 'Enoch', 'Enos', 'Ephah', 'Epher', 'Ephra', 'Ephraim', 'Ephrath', 'Ephron', 'Er', 'Erech', 'Eri', 'Es', 'Esau', 'Escape', 'Esek', 'Eshban', 'Eshcol', 'Ethiopia', 'Euphrat', 'Euphrates', 'Eve', 'Even', 'Every', 'Except', 'Ezbon', 'Ezer', 'Fear', 'Feed', 'Fifteen', 'Fill', 'For', 'Forasmuch', 'Forgive', 'From', 'Fulfil', 'G', 'Gad', 'Gaham', 'Galeed', 'Gatam', 'Gather', 'Gaza', 'Gentiles', 'Gera', 'Gerar', 'Gershon', 'Get', 'Gether', 'Gihon', 'Gilead', 'Girgashites', 'Girgasite', 'Give', 'Go', 'God', 'Gomer', 'Gomorrah', 'Goshen', 'Guni', 'Hadad', 'Hadar', 'Hadoram', 'Hagar', 'Haggi', 'Hai', 'Ham', 'Hamathite', 'Hamor', 'Hamul', 'Hanoch', 'Happy', 'Haran', 'Hast', 'Haste', 'Have', 'Havilah', 'Hazarmaveth', 'Hazezontamar', 'Hazo', 'He', 'Hear', 'Heaven', 'Heber', 'Hebrew', 'Hebrews', 'Hebron', 'Hemam', 'Hemdan', 'Here', 'Hereby', 'Heth', 'Hezron', 'Hiddekel', 'Hinder', 'Hirah', 'His', 'Hitti', 'Hittite', 'Hittites', 'Hivite', 'Hobah', 'Hori', 'Horite', 'Horites', 'How', 'Hul', 'Huppim', 'Husham', 'Hushim', 'Huz', 'I', 'If', 'In', 'Irad', 'Iram', 'Is', 'Isa', 'Isaac', 'Iscah', 'Ishbak', 'Ishmael', 'Ishmeelites', 'Ishuah', 'Isra', 'Israel', 'Issachar', 'Isui', 'It', 'Ithran', 'Jaalam', 'Jabal', 'Jabbok', 'Jac', 'Jachin', 'Jacob', 'Jahleel', 'Jahzeel', 'Jamin', 'Japhe', 'Japheth', 'Jared', 'Javan', 'Jebusite', 'Jebusites', 'Jegarsahadutha', 'Jehovahjireh', 'Jemuel', 'Jerah', 'Jetheth', 'Jetur', 'Jeush', 'Jezer', 'Jidlaph', 'Jimnah', 'Job', 'Jobab', 'Jokshan', 'Joktan', 'Jordan', 'Joseph', 'Jubal', 'Judah', 'Judge', 'Judith', 'Kadesh', 'Kadmonites', 'Karnaim', 'Kedar', 'Kedemah', 'Kemuel', 'Kenaz', 'Kenites', 'Kenizzites', 'Keturah', 'Kiriathaim', 'Kirjatharba', 'Kittim', 'Know', 'Kohath', 'Kor', 'Korah', 'LO', 'LORD', 'Laban', 'Lahairoi', 'Lamech', 'Lasha', 'Lay', 'Leah', 'Lehabim', 'Lest', 'Let', 'Letushim', 'Leummim', 'Levi', 'Lie', 'Lift', 'Lo', 'Look', 'Lot', 'Lotan', 'Lud', 'Ludim', 'Luz', 'Maachah', 'Machir', 'Machpelah', 'Madai', 'Magdiel', 'Magog', 'Mahalaleel', 'Mahalath', 'Mahanaim', 'Make', 'Malchiel', 'Male', 'Mam', 'Mamre', 'Man', 'Manahath', 'Manass', 'Manasseh', 'Mash', 'Masrekah', 'Massa', 'Matred', 'Me', 'Medan', 'Mehetabel', 'Mehujael', 'Melchizedek', 'Merari', 'Mesha', 'Meshech', 'Mesopotamia', 'Methusa', 'Methusael', 'Methuselah', 'Mezahab', 'Mibsam', 'Mibzar', 'Midian', 'Midianites', 'Milcah', 'Mishma', 'Mizpah', 'Mizraim', 'Mizz', 'Moab', 'Moabites', 'Moreh', 'Moreover', 'Moriah', 'Muppim', 'My', 'Naamah', 'Naaman', 'Nahath', 'Nahor', 'Naphish', 'Naphtali', 'Naphtuhim', 'Nay', 'Nebajoth', 'Neither', 'Night', 'Nimrod', 'Nineveh', 'Noah', 'Nod', 'Not', 'Now', 'O', 'Obal', 'Of', 'Oh', 'Ohad', 'Omar', 'On', 'Onam', 'Onan', 'Only', 'Ophir', 'Our', 'Out', 'Padan', 'Padanaram', 'Paran', 'Pass', 'Pathrusim', 'Pau', 'Peace', 'Peleg', 'Peniel', 'Penuel', 'Peradventure', 'Perizzit', 'Perizzite', 'Perizzites', 'Phallu', 'Phara', 'Pharaoh', 'Pharez', 'Phichol', 'Philistim', 'Philistines', 'Phut', 'Phuvah', 'Pildash', 'Pinon', 'Pison', 'Potiphar', 'Potipherah', 'Put', 'Raamah', 'Rachel', 'Rameses', 'Rebek', 'Rebekah', 'Rehoboth', 'Remain', 'Rephaims', 'Resen', 'Return', 'Reu', 'Reub', 'Reuben', 'Reuel', 'Reumah', 'Riphath', 'Rosh', 'Sabtah', 'Sabtech', 'Said', 'Salah', 'Salem', 'Samlah', 'Sarah', 'Sarai', 'Saul', 'Save', 'Say', 'Se', 'Seba', 'See', 'Seeing', 'Seir', 'Sell', 'Send', 'Sephar', 'Serah', 'Sered', 'Serug', 'Set', 'Seth', 'Shalem', 'Shall', 'Shalt', 'Shammah', 'Shaul', 'Shaveh', 'She', 'Sheba', 'Shebah', 'Shechem', 'Shed', 'Shel', 'Shelah', 'Sheleph', 'Shem', 'Shemeber', 'Shepho', 'Shillem', 'Shiloh', 'Shimron', 'Shinab', 'Shinar', 'Shobal', 'Should', 'Shuah', 'Shuni', 'Shur', 'Sichem', 'Siddim', 'Sidon', 'Simeon', 'Sinite', 'Sitnah', 'Slay', 'So', 'Sod', 'Sodom', 'Sojourn', 'Some', 'Spake', 'Speak', 'Spirit', 'Stand', 'Succoth', 'Surely', 'Swear', 'Syrian', 'Take', 'Tamar', 'Tarshish', 'Tebah', 'Tell', 'Tema', 'Teman', 'Temani', 'Terah', 'Thahash', 'That', 'The', 'Then', 'There', 'Therefore', 'These', 'They', 'Thirty', 'This', 'Thorns', 'Thou', 'Thus', 'Thy', 'Tidal', 'Timna', 'Timnah', 'Timnath', 'Tiras', 'To', 'Togarmah', 'Tola', 'Tubal', 'Tubalcain', 'Twelve', 'Two', 'Unstable', 'Until', 'Unto', 'Up', 'Upon', 'Ur', 'Uz', 'Uzal', 'We', 'What', 'When', 'Whence', 'Where', 'Whereas', 'Wherefore', 'Which', 'While', 'Who', 'Whose', 'Whoso', 'Why', 'Wilt', 'With', 'Woman', 'Ye', 'Yea', 'Yet', 'Zaavan', 'Zaphnathpaaneah', 'Zar', 'Zarah', 'Zeboiim', 'Zeboim', 'Zebul', 'Zebulun', 'Zemarite', 'Zepho', 'Zerah', 'Zibeon', 'Zidon', 'Zillah', 'Zilpah', 'Zimran', 'Ziphion', 'Zo', 'Zoar', 'Zohar', 'Zuzims', 'a', 'abated', 'abide', 'able', 'abode', 'abomination', 'about', 'above', 'abroad', 'absent', 'abundantly', 'accept', 'accepted', 'according', 'acknowledged', 'activity', 'add', 'adder', 'afar', 'afflict', 'affliction', 'afraid', 'after', 'afterward', 'afterwards', 'aga', 'again', 'against', 'age', 'aileth', 'air', 'al', 'alive', 'all', 'almon', 'alo', 'alone', 'aloud', 'also', 'altar', 'altogether', 'always', 'am', 'among', 'amongst', 'an', 'and', 'angel', 'angels', 'anger', 'angry', 'anguish', 'anointedst', 'anoth', 'another', 'answer', 'answered', 'any', 'anything', 'appe', 'appear', 'appeared', 'appease', 'appoint', 'appointed', 'aprons', 'archer', 'archers', 'are', 'arise', 'ark', 'armed', 'arms', 'army', 'arose', 'arrayed', 'art', 'artificer', 'as', 'ascending', 'ash', 'ashamed', 'ask', 'asked', 'asketh', 'ass', 'assembly', 'asses', 'assigned', 'asswaged', 'at', 'attained', 'audience', 'avenged', 'aw', 'awaked', 'away', 'awoke', 'back', 'backward', 'bad', 'bade', 'badest', 'badne', 'bak', 'bake', 'bakemeats', 'baker', 'bakers', 'balm', 'bands', 'bank', 'bare', 'barr', 'barren', 'basket', 'baskets', 'battle', 'bdellium', 'be', 'bear', 'beari', 'bearing', 'beast', 'beasts', 'beautiful', 'became', 'because', 'become', 'bed', 'been', 'befall', 'befell', 'before', 'began', 'begat', 'beget', 'begettest', 'begin', 'beginning', 'begotten', 'beguiled', 'beheld', 'behind', 'behold', 'being', 'believed', 'belly', 'belong', 'beneath', 'bereaved', 'beside', 'besides', 'besought', 'best', 'betimes', 'better', 'between', 'betwixt', 'beyond', 'binding', 'bird', 'birds', 'birthday', 'birthright', 'biteth', 'bitter', 'blame', 'blameless', 'blasted', 'bless', 'blessed', 'blesseth', 'blessi', 'blessing', 'blessings', 'blindness', 'blood', 'blossoms', 'bodies', 'boldly', 'bondman', 'bondmen', 'bondwoman', 'bone', 'bones', 'book', 'booths', 'border', 'borders', 'born', 'bosom', 'both', 'bottle', 'bou', 'boug', 'bough', 'bought', 'bound', 'bow', 'bowed', 'bowels', 'bowing', 'boys', 'bracelets', 'branches', 'brass', 'bre', 'breach', 'bread', 'breadth', 'break', 'breaketh', 'breaking', 'breasts', 'breath', 'breathed', 'breed', 'brethren', 'brick', 'brimstone', 'bring', 'brink', 'broken', 'brook', 'broth', 'brother', 'brought', 'brown', 'bruise', 'budded', 'build', 'builded', 'built', 'bulls', 'bundle', 'bundles', 'burdens', 'buried', 'burn', 'burning', 'burnt', 'bury', 'buryingplace', 'business', 'but', 'butler', 'butlers', 'butlership', 'butter', 'buy', 'by', 'cakes', 'calf', 'call', 'called', 'came', 'camel', 'camels', 'camest', 'can', 'cannot', 'canst', 'captain', 'captive', 'captives', 'carcases', 'carried', 'carry', 'cast', 'castles', 'catt', 'cattle', 'caught', 'cause', 'caused', 'cave', 'cease', 'ceased', 'certain', 'certainly', 'chain', 'chamber', 'change', 'changed', 'changes', 'charge', 'charged', 'chariot', 'chariots', 'chesnut', 'chi', 'chief', 'child', 'childless', 'childr', 'children', 'chode', 'choice', 'chose', 'circumcis', 'circumcise', 'circumcised', 'citi', 'cities', 'city', 'clave', 'clean', 'clear', 'cleave', 'clo', 'closed', 'clothed', 'clothes', 'cloud', 'clusters', 'co', 'coat', 'coats', 'coffin', 'cold', 'colours', 'colt', 'colts', 'come', 'comest', 'cometh', 'comfort', 'comforted', 'comi', 'coming', 'command', 'commanded', 'commanding', 'commandment', 'commandments', 'commended', 'committed', 'commune', 'communed', 'communing', 'company', 'compassed', 'compasseth', 'conceal', 'conceive', 'conceived', 'conception', 'concerning', 'concubi', 'concubine', 'concubines', 'confederate', 'confound', 'consent', 'conspired', 'consume', 'consumed', 'content', 'continually', 'continued', 'cool', 'corn', 'corrupt', 'corrupted', 'couch', 'couched', 'couching', 'could', 'counted', 'countenance', 'countries', 'country', 'covenant', 'covered', 'covering', 'created', 'creature', 'creepeth', 'creeping', 'cried', 'crieth', 'crown', 'cru', 'cruelty', 'cry', 'cubit', 'cubits', 'cunning', 'cup', 'current', 'curse', 'cursed', 'curseth', 'custom', 'cut', 'd', 'da', 'dainties', 'dale', 'damsel', 'damsels', 'dark', 'darkne', 'darkness', 'daughers', 'daught', 'daughte', 'daughter', 'daughters', 'day', 'days', 'dea', 'dead', 'deal', 'dealt', 'dearth', 'death', 'deceitfully', 'deceived', 'deceiver', 'declare', 'decreased', 'deed', 'deeds', 'deep', 'deferred', 'defiled', 'defiledst', 'delight', 'deliver', 'deliverance', 'delivered', 'denied', 'depart', 'departed', 'departing', 'deprived', 'descending', 'desire', 'desired', 'desolate', 'despised', 'destitute', 'destroy', 'destroyed', 'devour', 'devoured', 'dew', 'did', 'didst', 'die', 'died', 'digged', 'dignity', 'dim', 'dine', 'dipped', 'direct', 'discern', 'discerned', 'discreet', 'displease', 'displeased', 'distress', 'distressed', 'divide', 'divided', 'divine', 'divineth', 'do', 'doe', 'doer', 'doest', 'doeth', 'doing', 'dominion', 'done', 'door', 'dost', 'doth', 'double', 'doubled', 'doubt', 'dove', 'down', 'dowry', 'drank', 'draw', 'dread', 'dreadful', 'dream', 'dreamed', 'dreamer', 'dreams', 'dress', 'dressed', 'drew', 'dried', 'drink', 'drinketh', 'drinking', 'driven', 'drought', 'drove', 'droves', 'drunken', 'dry', 'duke', 'dukes', 'dunge', 'dungeon', 'dust', 'dwe', 'dwell', 'dwelled', 'dwelling', 'dwelt', 'e', 'ea', 'each', 'ear', 'earing', 'early', 'earring', 'earrings', 'ears', 'earth', 'east', 'eastward', 'eat', 'eaten', 'eatest', 'edge', 'eight', 'eighteen', 'eighty', 'either', 'elder', 'elders', 'eldest', 'eleven', 'else', 'embalm', 'embalmed', 'embraced', 'emptied', 'empty', 'end', 'ended', 'endued', 'endure', 'enemies', 'enlarge', 'enmity', 'enough', 'enquire', 'enter', 'entered', 'entreated', 'envied', 'erected', 'errand', 'escape', 'escaped', 'espied', 'establish', 'established', 'ev', 'even', 'evening', 'eventide', 'ever', 'everlasting', 'every', 'evil', 'ewe', 'ewes', 'exceeding', 'exceedingly', 'excel', 'excellency', 'except', 'exchange', 'experience', 'ey', 'eyed', 'eyes', 'fa', 'face', 'faces', 'fai', 'fail', 'failed', 'faileth', 'fainted', 'fair', 'fall', 'fallen', 'falsely', 'fame', 'families', 'famine', 'famished', 'far', 'fashion', 'fast', 'fat', 'fatfleshed', 'fath', 'fathe', 'father', 'fathers', 'fatness', 'faults', 'favour', 'favoured', 'fear', 'feared', 'fearest', 'feast', 'fed', 'feeble', 'feebler', 'feed', 'feeding', 'feel', 'feet', 'fell', 'fellow', 'felt', 'fema', 'female', 'fetch', 'fetched', 'fetcht', 'few', 'fie', 'field', 'fierce', 'fifteen', 'fifth', 'fifty', 'fig', 'fill', 'filled', 'find', 'findest', 'findeth', 'finding', 'fine', 'finish', 'finished', 'fir', 'fire', 'firmame', 'firmament', 'first', 'firstborn', 'firstlings', 'fish', 'fishes', 'five', 'flaming', 'fle', 'fled', 'fleddest', 'flee', 'flesh', 'flo', 'floc', 'flock', 'flocks', 'flood', 'floor', 'fly', 'fo', 'foal', 'foals', 'folk', 'follow', 'followed', 'following', 'folly', 'food', 'foolishly', 'foot', 'for', 'forbid', 'force', 'ford', 'foremost', 'foreskin', 'forgat', 'forget', 'forgive', 'forgotten', 'form', 'formed', 'former', 'forth', 'forty', 'forward', 'fou', 'found', 'fountain', 'fountains', 'four', 'fourscore', 'fourteen', 'fourteenth', 'fourth', 'fowl', 'fowls', 'freely', 'friend', 'friends', 'fro', 'from', 'frost', 'fruit', 'fruitful', 'fruits', 'fugitive', 'fulfilled', 'full', 'furnace', 'furniture', 'fury', 'gard', 'garden', 'garmen', 'garment', 'garments', 'gat', 'gate', 'gather', 'gathered', 'gathering', 'gave', 'gavest', 'generatio', 'generation', 'generations', 'get', 'getting', 'ghost', 'giants', 'gift', 'gifts', 'give', 'given', 'giveth', 'giving', 'glory', 'go', 'goa', 'goat', 'goats', 'gods', 'goest', 'goeth', 'going', 'gold', 'golden', 'gone', 'good', 'goodly', 'goods', 'gopher', 'got', 'gotten', 'governor', 'gr', 'grace', 'gracious', 'graciously', 'grap', 'grapes', 'grass', 'grave', 'gray', 'gre', 'great', 'greater', 'greatly', 'green', 'grew', 'grief', 'grieved', 'grievous', 'grisl', 'grisled', 'gro', 'ground', 'grove', 'grow', 'grown', 'guard', 'guiding', 'guiltiness', 'guilty', 'gutters', 'h', 'ha', 'habitations', 'had', 'hadst', 'hairs', 'hairy', 'half', 'halted', 'han', 'hand', 'handfuls', 'handle', 'handmaid', 'handmaidens', 'handmaids', 'hands', 'hang', 'hanged', 'hard', 'hardly', 'harlot', 'harm', 'harp', 'harvest', 'hast', 'haste', 'hasted', 'hastened', 'hastily', 'hate', 'hated', 'hath', 'have', 'haven', 'having', 'hazel', 'he', 'head', 'heads', 'healed', 'health', 'heap', 'hear', 'heard', 'hearken', 'hearkened', 'heart', 'hearth', 'hearts', 'heat', 'heav', 'heaven', 'heavens', 'heed', 'heel', 'heels', 'heifer', 'height', 'heir', 'held', 'help', 'hence', 'henceforth', 'her', 'herb', 'herd', 'herdmen', 'herds', 'here', 'herein', 'herself', 'hid', 'hide', 'high', 'hil', 'hills', 'him', 'himself', 'hind', 'hindermost', 'hire', 'hired', 'his', 'hith', 'hither', 'hold', 'hollow', 'home', 'honey', 'honour', 'honourable', 'hor', 'horror', 'horse', 'horsemen', 'horses', 'host', 'hotly', 'hou', 'hous', 'house', 'household', 'households', 'how', 'hundred', 'hundredfo', 'hundredth', 'hunt', 'hunter', 'hunting', 'hurt', 'husba', 'husband', 'husbandman', 'if', 'ill', 'image', 'images', 'imagination', 'imagined', 'in', 'increase', 'increased', 'indeed', 'inhabitants', 'inhabited', 'inherit', 'inheritance', 'iniquity', 'inn', 'innocency', 'instead', 'instructor', 'instruments', 'integrity', 'interpret', 'interpretation', 'interpretations', 'interpreted', 'interpreter', 'into', 'intreat', 'intreated', 'ir', 'is', 'isles', 'issue', 'it', 'itself', 'jewels', 'joined', 'joint', 'journey', 'journeyed', 'journeys', 'jud', 'judge', 'judged', 'judgment', 'just', 'justice', 'keep', 'keeper', 'kept', 'ki', 'kid', 'kids', 'kill', 'killed', 'kind', 'kindled', 'kindly', 'kindness', 'kindred', 'kinds', 'kine', 'king', 'kingdom', 'kings', 'kiss', 'kissed', 'kn', 'knead', 'kneel', 'knees', 'knew', 'knife', 'know', 'knowest', 'knoweth', 'knowing', 'knowledge', 'known', 'la', 'labour', 'lack', 'lad', 'ladder', 'lade', 'laded', 'laden', 'lads', 'laid', 'lamb', 'lambs', 'lamentati', 'lamp', 'lan', 'land', 'lands', 'language', 'large', 'last', 'laugh', 'laughed', 'law', 'lawgiver', 'laws', 'lay', 'lead', 'leaf', 'lean', 'leanfleshed', 'leap', 'leaped', 'learned', 'least', 'leave', 'leaves', 'led', 'left', 'length', 'lentiles', 'lesser', 'lest', 'let', 'li', 'lie', 'lien', 'liest', 'lieth', 'life', 'lift', 'lifted', 'light', 'lighted', 'lightly', 'lights', 'like', 'likene', 'likeness', 'linen', 'lingered', 'lion', 'little', 'live', 'lived', 'lives', 'liveth', 'living', 'lo', 'lodge', 'lodged', 'loins', 'long', 'longedst', 'longeth', 'look', 'looked', 'loose', 'lord', 'lords', 'loss', 'loud', 'love', 'loved', 'lovest', 'loveth', 'lower', 'lying', 'm', 'ma', 'made', 'magicians', 'magnified', 'maid', 'maiden', 'maidservants', 'make', 'male', 'males', 'man', 'mandrakes', 'manner', 'many', 'mark', 'marriages', 'married', 'marry', 'marvelled', 'mast', 'master', 'matter', 'may', 'mayest', 'me', 'mead', 'meadow', 'meal', 'mean', 'meanest', 'meant', 'measures', 'meat', 'meditate', 'meet', 'meeteth', 'men', 'menservants', 'mention', 'merchant', 'merchantmen', 'mercies', 'merciful', 'mercy', 'merry', 'mess', 'messenger', 'messengers', 'messes', 'met', 'mi', 'midst', 'midwife', 'might', 'mightier', 'mighty', 'milch', 'milk', 'millions', 'mind', 'mine', 'mirth', 'mischief', 'mist', 'mistress', 'mock', 'mocked', 'mocking', 'money', 'month', 'months', 'moon', 'more', 'moreover', 'morever', 'morning', 'morrow', 'morsel', 'morter', 'most', 'mother', 'mou', 'mount', 'mountain', 'mountains', 'mourn', 'mourned', 'mourning', 'mouth', 'mouths', 'moved', 'moveth', 'moving', 'much', 'mules', 'multiplied', 'multiply', 'multiplying', 'multitude', 'must', 'my', 'myrrh', 'myself', 'n', 'na', 'naked', 'nakedness', 'name', 'named', 'names', 'nati', 'natio', 'nation', 'nations', 'nativity', 'ne', 'near', 'neck', 'needeth', 'needs', 'neither', 'never', 'next', 'nig', 'nigh', 'night', 'nights', 'nine', 'nineteen', 'ninety', 'no', 'none', 'noon', 'nor', 'north', 'northward', 'nostrils', 'not', 'nothing', 'nought', 'nourish', 'nourished', 'now', 'number', 'numbered', 'numbering', 'nurse', 'nuts', 'o', 'oa', 'oak', 'oath', 'obeisance', 'obey', 'obeyed', 'observed', 'obtain', 'occasion', 'occupation', 'of', 'off', 'offended', 'offer', 'offered', 'offeri', 'offering', 'offerings', 'office', 'officer', 'officers', 'oil', 'old', 'olive', 'on', 'one', 'ones', 'only', 'onyx', 'open', 'opened', 'openly', 'or', 'order', 'organ', 'oth', 'other', 'ou', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'overcome', 'overdrive', 'overseer', 'oversig', 'overspread', 'overtake', 'overthrew', 'overthrow', 'overtook', 'own', 'oxen', 'parcel', 'part', 'parted', 'parts', 'pass', 'passed', 'past', 'pasture', 'path', 'pea', 'peace', 'peaceable', 'peaceably', 'peop', 'people', 'peradventure', 'perceived', 'perfect', 'perform', 'perish', 'perpetual', 'person', 'persons', 'physicians', 'piece', 'pieces', 'pigeon', 'pilgrimage', 'pillar', 'pilled', 'pillows', 'pit', 'pitch', 'pitched', 'pitcher', 'pla', 'place', 'placed', 'places', 'plagued', 'plagues', 'plain', 'plains', 'plant', 'planted', 'played', 'pleasant', 'pleased', 'pleaseth', 'pleasure', 'pledge', 'plenteous', 'plenteousness', 'plenty', 'pluckt', 'point', 'poor', 'poplar', 'portion', 'possess', 'possessi', 'possession', 'possessions', 'possessor', 'posterity', 'pottage', 'poured', 'poverty', 'pow', 'power', 'praise', 'pray', 'prayed', 'precious', 'prepared', 'presence', 'present', 'presented', 'preserve', 'preserved', 'pressed', 'prevail', 'prevailed', 'prey', 'priest', 'priests', 'prince', 'princes', 'pris', 'prison', 'prisoners', 'proceedeth', 'process', 'profit', 'progenitors', 'prophet', 'prosper', 'prospered', 'prosperous', 'protest', 'proved', 'provender', 'provide', 'provision', 'pulled', 'punishment', 'purchase', 'purchased', 'purposing', 'pursue', 'pursued', 'put', 'putting', 'quart', 'quickly', 'quite', 'quiver', 'raiment', 'rain', 'rained', 'raise', 'ram', 'rams', 'ran', 'rank', 'raven', 'ravin', 'reach', 'reached', 'ready', 'reason', 'rebelled', 'rebuked', 'receive', 'received', 'red', 'redeemed', 'refrain', 'refrained', 'refused', 'regard', 'reign', 'reigned', 'remained', 'remaineth', 'remember', 'remembered', 'remove', 'removed', 'removing', 'renown', 'rent', 'repented', 'repenteth', 'replenish', 'report', 'reproa', 'reproach', 'reproved', 'require', 'required', 'requite', 'reserved', 'respect', 'rest', 'rested', 'restore', 'restored', 'restrained', 'return', 'returned', 'reviv', 'reward', 'rewarded', 'ri', 'rib', 'ribs', 'rich', 'riches', 'rid', 'ride', 'rider', 'right', 'righteous', 'righteousness', 'rightly', 'ring', 'ringstraked', 'ripe', 'rise', 'risen', 'riv', 'river', 'rode', 'rods', 'roll', 'rolled', 'roof', 'room', 'rooms', 'rose', 'roughly', 'round', 'rouse', 'royal', 'rul', 'rule', 'ruled', 'ruler', 'rulers', 'run', 's', 'sa', 'sac', 'sack', 'sackcloth', 'sacks', 'sacrifice', 'sacrifices', 'sad', 'saddled', 'sadly', 'said', 'saidst', 'saith', 'sake', 'sakes', 'salt', 'salvation', 'same', 'sanctified', 'sand', 'sat', 'save', 'saved', 'saving', 'savour', 'savoury', 'saw', 'sawest', 'say', 'saying', 'scarce', 'scarlet', 'scatter', 'scattered', 'sceptre', 'sea', 'searched', 'seas', 'season', 'seasons', 'second', 'secret', 'secretly', 'see', 'seed', 'seedtime', 'seeing', 'seek', 'seekest', 'seem', 'seemed', 'seen', 'seest', 'seeth', 'selfsame', 'selfwill', 'sell', 'send', 'sent', 'separate', 'separated', 'sepulchre', 'sepulchres', 'serpent', 'serva', 'servan', 'servant', 'servants', 'serve', 'served', 'service', 'set', 'seven', 'sevenfold', 'sevens', 'seventeen', 'seventeenth', 'seventh', 'seventy', 'sewed', 'sh', 'shadow', 'shall', 'shalt', 'shamed', 'shaved', 'she', 'sheaf', 'shear', 'sheaves', 'shed', 'sheddeth', 'sheep', 'sheepshearers', 'shekel', 'shekels', 'shepherd', 'shepherds', 'shew', 'shewed', 'sheweth', 'shield', 'ships', 'shoelatchet', 'shore', 'shortly', 'shot', 'should', 'shoulder', 'shoulders', 'shouldest', 'shrank', 'shrubs', 'shut', 'si', 'side', 'sight', 'signet', 'signs', 'silv', 'silver', 'sin', 'since', 'sinew', 'sinners', 'sinning', 'sir', 'sist', 'sister', 'sit', 'six', 'sixteen', 'sixth', 'sixty', 'skins', 'slain', 'slaughter', 'slay', 'slayeth', 'sle', 'sleep', 'slept', 'slew', 'slime', 'slimepits', 'small', 'smell', 'smelled', 'smite', 'smoke', 'smoking', 'smooth', 'smote', 'so', 'sod', 'softly', 'sojourn', 'sojourned', 'sojourner', 'sold', 'sole', 'solemnly', 'some', 'son', 'songs', 'sons', 'soon', 'sore', 'sorely', 'sorrow', 'sort', 'sou', 'sought', 'soul', 'souls', 'south', 'southward', 'sow', 'sowed', 'space', 'spake', 'spare', 'spe', 'speak', 'speaketh', 'speaking', 'speckl', 'speckled', 'spee', 'speech', 'speed', 'speedily', 'spent', 'spi', 'spicery', 'spices', 'spies', 'spilled', 'spirit', 'spoil', 'spoiled', 'spoken', 'sporting', 'spotted', 'spread', 'springing', 'sprung', 'staff', 'stalk', 'stand', 'standest', 'stars', 'state', 'statutes', 'stay', 'stayed', 'ste', 'stead', 'steal', 'steward', 'still', 'stink', 'sto', 'stole', 'stolen', 'stone', 'stones', 'stood', 'stooped', 'stopped', 'store', 'storehouses', 'stories', 'straitly', 'strakes', 'strange', 'stranger', 'strangers', 'straw', 'street', 'strength', 'strengthened', 'stretched', 'stricken', 'strife', 'stript', 'strive', 'strong', 'stronger', 'strove', 'struggled', 'stuff', 'subdue', 'submit', 'substance', 'subtil', 'subtilty', 'such', 'suck', 'suffered', 'summer', 'sun', 'supplanted', 'sure', 'surely', 'surety', 'sustained', 'sware', 'swear', 'sweat', 'sweet', 'sword', 'sworn', 'tabret', 'tak', 'take', 'taken', 'talked', 'talking', 'tar', 'tarried', 'tarry', 'teeth', 'tell', 'tempt', 'ten', 'tender', 'tenor', 'tent', 'tenth', 'tents', 'terror', 'th', 'than', 'that', 'the', 'thee', 'their', 'them', 'themselv', 'themselves', 'then', 'thence', 'there', 'thereby', 'therefore', 'therein', 'thereof', 'thereon', 'these', 'they', 'thi', 'thicket', 'thigh', 'thin', 'thine', 'thing', 'things', 'think', 'third', 'thirteen', 'thirteenth', 'thirty', 'this', 'thistles', 'thither', 'thoroughly', 'those', 'thou', 'though', 'thought', 'thoughts', 'thousand', 'thousands', 'thread', 'three', 'threescore', 'threshingfloor', 'throne', 'through', 'throughout', 'thus', 'thy', 'thyself', 'tidings', 'till', 'tiller', 'tillest', 'tim', 'time', 'times', 'tithes', 'to', 'togeth', 'together', 'toil', 'token', 'told', 'tongue', 'tongues', 'too', 'took', 'top', 'tops', 'torn', 'touch', 'touched', 'toucheth', 'touching', 'toward', 'tower', 'towns', 'tr', 'trade', 'traffick', 'trained', 'travail', 'travailed', 'treasure', 'tree', 'trees', 'trembled', 'trespass', 'tribes', 'tribute', 'troop', 'troubled', 'trough', 'troughs', 'tru', 'true', 'truly', 'truth', 'turn', 'turned', 'turtledove', 'twel', 'twelve', 'twentieth', 'twenty', 'twice', 'twins', 'two', 'unawares', 'uncircumcised', 'uncovered', 'under', 'understand', 'understood', 'ungirded', 'unit', 'unleavened', 'until', 'unto', 'up', 'upon', 'uppermost', 'upright', 'upward', 'urged', 'us', 'utmost', 'vagabond', 'vail', 'vale', 'valley', 'vengeance', 'venison', 'verified', 'verily', 'very', 'vessels', 'vestures', 'victuals', 'vine', 'vineyard', 'violence', 'violently', 'virgin', 'vision', 'visions', 'visit', 'visited', 'voi', 'voice', 'void', 'vow', 'vowed', 'vowedst', 'w', 'wa', 'wages', 'wagons', 'waited', 'walk', 'walked', 'walketh', 'walking', 'wall', 'wander', 'wandered', 'wandering', 'war', 'ward', 'was', 'wash', 'washed', 'wast', 'wat', 'watch', 'water', 'watered', 'watering', 'waters', 'waxed', 'waxen', 'way', 'ways', 'we', 'wealth', 'weaned', 'weapons', 'wearied', 'weary', 'week', 'weep', 'weig', 'weighed', 'weight', 'welfare', 'well', 'wells', 'went', 'wentest', 'wept', 'were', 'west', 'westwa', 'whales', 'what', 'whatsoever', 'wheat', 'whelp', 'when', 'whence', 'whensoever', 'where', 'whereby', 'wherefore', 'wherein', 'whereof', 'whereon', 'wherewith', 'whether', 'which', 'while', 'white', 'whither', 'who', 'whole', 'whom', 'whomsoever', 'whoredom', 'whose', 'whosoever', 'why', 'wi', 'wick', 'wicked', 'wickedly', 'wickedness', 'widow', 'widowhood', 'wife', 'wild', 'wilderness', 'will', 'willing', 'wilt', 'wind', 'window', 'windows', 'wine', 'winged', 'winter', 'wise', 'wit', 'with', 'withered', 'withheld', 'withhold', 'within', 'without', 'witness', 'wittingly', 'wiv', 'wives', 'wo', 'wolf', 'woman', 'womb', 'wombs', 'women', 'womenservan', 'womenservants', 'wondering', 'wood', 'wor', 'word', 'words', 'work', 'worse', 'worship', 'worshipped', 'worth', 'worthy', 'wot', 'wotteth', 'would', 'wouldest', 'wounding', 'wrapped', 'wrath', 'wrestled', 'wrestlings', 'wrong', 'wroth', 'wrought', 'y', 'ye', 'yea', 'year', 'yearn', 'years', 'yesternight', 'yet', 'yield', 'yielded', 'yielding', 'yoke', 'yonder', 'you', 'young', 'younge', 'younger', 'youngest', 'your', 'yourselves', 'youth']
```
```python
# Entrada
len(set(text3))
```

```python
# Salida
2789
```

```python
# Entrada
len(set(text3)) / len(text3)
```

```python
# Salida
0.06230453042623537
```

```python
# Entrada
text3.count("smote")
```

```python
# Salida
5
```

```python
# Entrada
100 * text4.count('a') / len(text4)
```

```python
# Salida
1.4569256756756757
```
```python
# Entrada
lexical_diversity(text3)
```

```python
# Salida
0.06230453042623537
```

```python
# Entrada
lexical_diversity(text5)
```

```python
# Salida
0.13477005109975562
```

```python
# Entrada
percentage(4, 5)
```

```python
# Salida
0.80
```
```python
# Entrada
percentage(text4.count('a'), len(text4))
```

```python
# Salida
1.4643016433938312
```

## Ejercicios "Your turn"

### 1. Enter a few more expressions of your own. You can use asterisk (*) for multiplication and slash (/) for division, and parentheses for bracketing expressions.

```python
# Expresiones matemáticas
(5 + 3) * 2
10 / (2 + 3)
7 * (4 - 1) + 6 / 2
```
```python
# Salida de los comandos
16.0
2.0
24.0
```

### 2. Try searching for other words; to save re-typing, you might be able to use up-arrow, Ctrl-up-arrow or Alt-p to access the previous command and modify the word being searched.
You can also try searches on some of the other texts we have included. For example, search Sense and Sensibility for the word affection, using text2.concordance("affection"). Search the book of Genesis to find out how long some people lived, using text3.concordance("lived"). You could look at text4, the Inaugural Address Corpus, to see examples of English going back to 1789, and search for words like nation, terror, god to see how these words have been used differently over time. We've also included text5, the NPS Chat Corpus: search this for unconventional words like im, ur, lol. (Note that this corpus is uncensored!)

```python
# Búsquedas en textos de NLTK Book
from nltk.book import *
text2
text2.concordance("affection")
text3
text3.concordance("lived")
text4
text3.concordance("nation")
text3.concordance("god")
text5
text5.concordance("lol")
```
```python
# Salida de los comandos
*** Introductory Examples for the NLTK Book ***
Loading text1, ..., text9 and sent1, ..., sent9
Type the name of the text or sentence to view it.
Type: 'texts()' or 'sents()' to list the materials.
text1: Moby Dick by Herman Melville 1851
text2: Sense and Sensibility by Jane Austen 1811
text3: The Book of Genesis
text4: Inaugural Address Corpus
text5: Chat Corpus
text6: Monty Python and the Holy Grail
text7: Wall Street Journal
text8: Personals Corpus
text9: The Man Who Was Thursday by G . K . Chesterton 1908

<Text: Sense and Sensibility by Jane Austen 1811>
Displaying 25 of 79 matches:
, however , and , as a mark of his affection for the three girls , he left them
t . It was very well known that no affection was ever supposed to exist between
deration of politeness or maternal affection on the side of the former , the tw
d the suspicion -- the hope of his affection for me may warrant , without impru
hich forbade the indulgence of his affection . She knew that his mother neither
rd she gave one with still greater affection . Though her late conversation wit
 can never hope to feel or inspire affection again , and if her home be uncomfo
m of the sense , elegance , mutual affection , and domestic comfort of the fami
, and which recommended him to her affection beyond every thing else . His soci
ween the parties might forward the affection of Mr . Willoughby , an equally st
 the most pointed assurance of her affection . Elinor could not be surprised at
he natural consequence of a strong affection in a young and ardent mind . This 
 opinion . But by an appeal to her affection for her mother , by representing t
 every alteration of a place which affection had established as perfect with hi
e will always have one claim of my affection , which no other can possibly shar
f the evening declared at once his affection and happiness . " Shall we see you
ause he took leave of us with less affection than his usual behaviour has shewn
ness ." " I want no proof of their affection ," said Elinor ; " but of their en
onths , without telling her of his affection ;-- that they should part without 
ould be the natural result of your affection for her . She used to be all unres
distinguished Elinor by no mark of affection . Marianne saw and listened with i
th no inclination for expense , no affection for strangers , no profession , an
till distinguished her by the same affection which once she had felt no doubt o
al of her confidence in Edward ' s affection , to the remembrance of every mark
 was made ? Had he never owned his affection to yourself ?" " Oh , no ; but if 

<Text: The Book of Genesis>
Displaying 25 of 38 matches:
ay when they were created . And Adam lived an hundred and thirty years , and be
ughters : And all the days that Adam lived were nine hundred and thirty yea and
nd thirty yea and he died . And Seth lived an hundred and five years , and bega
ve years , and begat Enos : And Seth lived after he begat Enos eight hundred an
welve years : and he died . And Enos lived ninety years , and begat Cainan : An
 years , and begat Cainan : And Enos lived after he begat Cainan eight hundred 
ive years : and he died . And Cainan lived seventy years and begat Mahalaleel :
rs and begat Mahalaleel : And Cainan lived after he begat Mahalaleel eight hund
years : and he died . And Mahalaleel lived sixty and five years , and begat Jar
s , and begat Jared : And Mahalaleel lived after he begat Jared eight hundred a
and five yea and he died . And Jared lived an hundred sixty and two years , and
o years , and he begat Eno And Jared lived after he begat Enoch eight hundred y
 and two yea and he died . And Enoch lived sixty and five years , and begat Met
 ; for God took him . And Methuselah lived an hundred eighty and seven years , 
 , and begat Lamech . And Methuselah lived after he begat Lamech seven hundred 
nd nine yea and he died . And Lamech lived an hundred eighty and two years , an
ch the LORD hath cursed . And Lamech lived after he begat Noah five hundred nin
naan shall be his servant . And Noah lived after the flood three hundred and fi
xad two years after the flo And Shem lived after he begat Arphaxad five hundred
at sons and daughters . And Arphaxad lived five and thirty years , and begat Sa
ars , and begat Salah : And Arphaxad lived after he begat Salah four hundred an
begat sons and daughters . And Salah lived thirty years , and begat Eber : And 
y years , and begat Eber : And Salah lived after he begat Eber four hundred and
 begat sons and daughters . And Eber lived four and thirty years , and begat Pe
y years , and begat Peleg : And Eber lived after he begat Peleg four hundred an

<Text: Inaugural Address Corpus>
Displaying 8 of 8 matches:
 th And I will make of thee a great nation , and I will bless thee , and make 
 four hundred years ; And also that nation , whom they shall serve , will I ju
beget , and I will make him a great nation . But my covenant will I establish 
ll surely become a great and mighty nation , and all the nations of the earth 
D , wilt thou slay also a righteous nation ? Said he not unto me , She is my s
 son of the bondwoman will I make a nation , because he is thy seed . And Abra
 hand ; for I will make him a great nation . And God opened her eyes , and she
ghty : be fruitful and multiply ; a nation and a company of nations shall be o

Displaying 25 of 231 matches:
                     In the beginning God created the heaven and the earth . An
 face of the deep . And the Spirit of God moved upon the face of the waters . A
ved upon the face of the waters . And God said , Let there be light : and there
 be light : and there was light . And God saw the light , that it was good : an
aw the light , that it was good : and God divided the light from the darkness .
ded the light from the darkness . And God called the light Day , and the darkne
 the morning were the first day . And God said , Let there be a firmament in th
vide the waters from the waters . And God made the firmament , and divided the 
above the firmame and it was so . And God called the firmament Heaven . And the
the morning were the second day . And God said , Let the waters under the heave
the dry land appe and it was so . And God called the dry land Earth ; and the g
gether of the waters called he Se and God saw that it was good . And God said ,
Se and God saw that it was good . And God said , Let the earth bring forth gras
seed was in itself , after his ki and God saw that it was good . And the evenin
 the morning were the third day . And God said , Let there be lights in the fir
ight upon the ear and it was so . And God made two great lights ; the greater l
 the nig he made the stars also . And God set them in the firmament of the heav
 divide the light from the darkne and God saw that it was good . And the evenin
the morning were the fourth day . And God said , Let the waters bring forth abu
in the open firmament of heaven . And God created great whales , and every livi
nd every winged fowl after his ki and God saw that it was good . And God blesse
ki and God saw that it was good . And God blessed them , saying , Be fruitful ,
 the morning were the fifth day . And God said , Let the earth bring forth the 
arth after his ki and it was so . And God made the beast of the earth after his
epeth upon the earth after his ki and God saw that it was good . And God said ,

<Text: Chat Corpus>
Displaying 25 of 822 matches:
ast PART 24 / m boo . 26 / m and sexy lol U115 boo . JOIN PART he drew a girl w
ope he didnt draw a penis PART ewwwww lol & a head between her legs JOIN JOIN s
a bowl i got a blunt an a bong ...... lol JOIN well , glad it worked out my cha
e " PART Hi U121 in ny . ACTION would lol @ U121 . . . but appearently she does
30 make sure u buy a nice ring for U6 lol U7 Hi U115 . ACTION isnt falling for 
 didnt ya hear !!!! PART JOIN geeshhh lol U6 PART hes deaf ppl here dont get it
es nobody here i wanna misbeahve with lol JOIN so read it . thanks U7 .. Im hap
ies want to chat can i talk to him !! lol U121 !!! forwards too lol JOIN ALL PE
k to him !! lol U121 !!! forwards too lol JOIN ALL PErvs ... redirect to U121 '
 loves ME the most i love myself JOIN lol U44 how do u know that what ? jerkett
ng wrong ... i can see it in his eyes lol U20 = fiance Jerketts lmao wtf yah I 
cooler by the minute what 'd I miss ? lol noo there too much work ! why not ?? 
 that mean I want you ? U6 hello room lol U83 and this .. has been the grammar 
 the rule he 's in PM land now though lol ah ok i wont bug em then someone wann
flight to hell :) lmao bbl maybe PART LOL lol U7 it was me , U83 hahah U83 ! 80
ht to hell :) lmao bbl maybe PART LOL lol U7 it was me , U83 hahah U83 ! 808265
082653953 K-Fed got his ass kicked .. Lol . ACTION laughs . i got a first class
 . i got a first class ticket to hell lol U7 JOIN any texas girls in here ? any
 . whats up U155 i was only kidding . lol he 's a douchebag . Poor U121 i 'm bo
 ??? sits with U30 Cum to my shower . lol U121 . ACTION U1370 watches his nads 
 ur nad with a stick . ca u U23 ewwww lol *sniffs* ewwwwww PART U115 ! owww spl
ACTION is resisting . ur female right lol U115 beeeeehave Remember the LAst tim
pm's me . charge that is 1.99 / min . lol @ innocent hahah lol .... yeah LOLOLO
 is 1.99 / min . lol @ innocent hahah lol .... yeah LOLOLOLLL U12 thats not nic
s . lmao no U115 Check my record . :) Lol lick em U7 U23 how old r u lol Way to
```

### 3. Pick another pair of words and compare their usage in two different texts, using the similar() and common_contexts() functions.

```python
# Contextos comunes
text2 
text2.common_contexts(["hope", "affection"])
```

```python
# Salida del comando
the_of of_. of_and
```

### 4. How many times does the word lol appear in text5? How much is this as a percentage of the total number of words in this text?

```python
# Frecuencia y porcentaje de "lol" en text5
text5
text5.count("lol")
len(set(text5)) / len(text5)
```

```python
# Salida de los comandos
<Text: Chat Corpus>
704
0.13477005109975562
```

## Ejercicios

### ☼ Try using the Python interpreter as a calculator, and typing expressions like 12 / (4 + 1).
```python
# Entradas
18 * 9 + 7 / (8 - 2)
2 * 4 + 7
4 + 8 + 2 + 9 * 3
```

```python
# Salida
163.16666666666666
15
41
```

### ☼ Given an alphabet of 26 letters, there are 26 to the power 10, or 26 ** 10, ten-letter strings we can form. That works out to 141167095653376. How many hundred-letter strings are possible?
```python
# Entrada
26 ** 100
```
```python
# Salida
3142930641582938830174357788501626427282669988762475256374173175398995908420104023465432599069702289330964075081611719197835869803511992549376
```

### ☼ The Python multiplication operation can be applied to lists. What happens when you type ['Monty', 'Python'] * 20, or 3 * sent1?

```python
# Entradas
['Monty', 'Python'] * 20
3 * sent1
```

```python
# Salida
['Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python', 'Monty', 'Python']
['Call', 'me', 'Ishmael', '.', 'Call', 'me', 'Ishmael', '.', 'Call', 'me', 'Ishmael', '.']
```

### ☼ Review 1 on computing with language. How many words are there in text2? How many distinct words are there?
 ```python
# Entrada
text2
len(text2)
len(set(text2))
```
```python
# Salida
<Text: Sense and Sensibility by Jane Austen 1811>
141576
6598
```

### ☼ Compare the lexical diversity scores for humor and romance fiction in 1.1. Which genre is more lexically diverse?

Humor es más diverso léxicamente que romance. Superando a romance en diversidad léxica por aproximadamente 12%.

### ☼ Produce a dispersion plot of the four main protagonists in Sense and Sensibility: Elinor, Marianne, Edward, and Willoughby. What can you observe about the different roles played by the males and females in this novel? Can you identify the couples?
```python
text2
text2.dispersion_plot(["Elinor", "Marianne", "Edward", "Willoughby"])
```

### ☼ Find the collocations in text5.
```python
# Entrada
text5
text5.collocations()
```

```python
# Salida
wanna chat; PART JOIN; MODE #14-19teens; JOIN PART; PART PART;
cute.-ass MP3; MP3 player; JOIN JOIN; times .. .; ACTION watches; guys
wanna; song lasts; last night; ACTION sits; -...)...- S.M.R.; Lime
Player; Player 12%; dont know; lez gurls; long time
```


### ☼ Consider the following Python expression: len(set(text4)). State the purpose of this expression. Describe the two steps involved in performing this computation
set(text4) — convierte la secuencia de tokens en un conjunto, eliminando duplicados (queda cada token una sola vez).

len() — cuenta cuántos elementos únicos hay en ese conjunto.