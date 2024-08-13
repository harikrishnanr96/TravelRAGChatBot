import re
from pypdf import PdfReader
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import textacy
import textacy.preprocessing as tprep


reader = PdfReader('london.pdf')

print(len(reader.pages))

pages = reader.pages
text = ""
for i in range(3,394):
    page = reader.pages[i]
    text += page.extract_text()

def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.replace('£','pounds ')
    return text
# print(text)
# Remove punctuation
# text = text.translate(str.maketrans('', '', string.punctuation))
# Convert to lowercase
text = ' '.join(text.split())

textr= normalize(text)
texts= re.sub(r'Lonely\s*Planet', '', textr, flags=re.IGNORECASE)

# text = text.lower()
# print(text)
with open('travel.txt','w',encoding="utf-8") as f:
    f.write(texts)
