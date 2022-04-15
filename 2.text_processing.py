with open('/content/drive/My Drive/PhD/data/amazon_new.json', "r", encoding="utf8") as f: #change what in '...' with your direction of data
   data = json.load(f)
data_return = pd.DataFrame(data)
# Remove empty reviews
data_return =  data_return.loc[~((data_return.text.isna() == True) & (data_return.title.isna() == True))].reset_index(drop=True)
# Drop duplications
data_return = data_return.drop_duplicates(subset=['text','id'], keep='last').reset_index(drop=True)

#clean reviews

stop_words = list(set(stopwords.words('english')))
def remove_stops(line):
  line = [w for w in line if not w in stops]
  return line
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}
contraction_mapping2 = { k.replace("'", "’"): v for k, v in contraction_mapping.items() }

def clean_format(line):
    line = line.lower()
    line = re.sub("\n"," ",line)
    line = re.sub("\t"," ",line)
    line = re.sub("\r"," ",line)
    line = re.sub("-"," ",line)
    line = re.sub(r"'s\b"," ", line)
    #map contraction 
    line = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in line.split(" ")])
    line = ' '.join([contraction_mapping2[t] if t in contraction_mapping2 else t for t in line.split(" ")])
    #first have to remove "," or "."
    re_punc = re.compile('[%s]' % re.escape('\.\,®•�!#"\'&\()*+/:;<=>?@[\\]^_`{|}~'))
    # remove punctuation from each word
    line = re_punc.sub(' ', line)
    # strip multiple spaces
    line = ' '.join([word for word in line.split(' ') if word.isalpha()])

    stripped_text = re.sub(' +',' ', line)
    stripped_text = stripped_text.lstrip().rstrip()
    #spelling check
    textBlb = TextBlob(stripped_text)
    stripped_text = str(textBlb.correct())
    return stripped_text

#cross check summary and text review to see any missing value
text_na = data_return['text'].loc[data_return.text.isna() == True]
text = data_return.text

for i in text_na.index.values:
    if data_return['title'].iloc[i] is not np.nan and data_return['text'] is not np.nan:
      data_return['text'].iloc[i] = data_return['text'].iloc[i] + ' ' + data_return['title'].iloc[i]
    else: pass

for i, line in enumerate(data_return['text']):
    if line in ['None','N/A','nan','NAN','n/a','NA','na','',' ']:
        print(line)
        data_return['text'][i] = np.nan
    else: pass 
text_cleaned = pd.Series([clean_format(line) for line in tqdm(text.values)])
for i, line in enumerate(text_cleaned):
    if line in ['None','N/A','nan','NAN','n/a','NA','na','',' ']:
        print(line)
        text_cleaned[i] = np.nan
    else: pass 
data_return['text_cleaned_phase1'] = text_cleaned
data_return = data_return.dropna(subset= ['text_cleaned_phase1'], axis=0).reset_index(drop=True)
reviews = data_return.text_cleaned_phase1.reset_index(drop=True)


# Cleaning text data using Gensim preprocessing tool
import gensim
data = list(reviews)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

#Build Bigram & Trigram models
data_words = list(sent_to_words(data))
print(data_words[:1])
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=5) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=20) 
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[100]]])

%%time
import nltk
nltk.download('stopwords')

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in tqdm(texts)]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in tqdm(texts)]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in tqdm(texts)]

def lemmatization(texts, allowed_postags=['NOUN','ADJ', 'VERB','PROPN','NUM','ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
from gensim.utils import simple_preprocess
import spacy

# NLTK Stop words
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
  
stop_words = stopwords.words('english')
stop_words.extend(['would','could'])
# python3 -m spacy download en

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN','ADJ', 'VERB','PROPN','NUM','ADV'])

# Form Bigrams
data_words_bigrams = make_bigrams(data_lemmatized)
# Form Trigrams

data_words_trigrams = make_trigrams(data_words_bigrams)

#Check potential missing values after filtering our stopwords
data_words_trigrams  = [' '.join(doc) for doc in data_words_trigrams]
data_return['data_words_trigrams'] = data_words_trigrams
for i, line in enumerate(data_return['data_words_trigrams']):
    if line in ['None','N/A','nan','NAN','n/a','NA','na','',' ']:
        print(line)
        data_return['data_words_trigrams'][i] = np.nan
    else: pass 
data_return = data_return.dropna(subset=['data_words_trigrams'], axis=0).reset_index(drop=True)
data_words_trigrams = [ doc.split(' ') for doc in data_return.data_words_trigrams]
data = list(data_return.text)

#Check reviews that were already verified by Amazon
data_return_verified = data_return.loc[ data_return.verified == 'Verified Purchase' ]


#Save final cleaned reviews from Amazon
data_return_verified.to_csv('/content/drive/My Drive/PhD/JOM_data/data_return_verified.csv')

#Save cleaned dataset
data_return.to_csv('/content/drive/My Drive/PhD/JOM_data/data_return_clean_phase1.json')

food = data_return_verified.loc[data_return_verified['key'] == 'Food & Grocery']
book = data_return_verified.loc[data_return_verified['key'] == 'Books']
fashion = data_return_verified.loc[data_return_verified['key'] == 'Clothes']
elec = data_return_verified.loc[data_return_verified['key'] == 'Electronic Device']
home = data_return_verified.loc[data_return_verified['key'] == 'Home Appliance']
data_lemmatized = [row.split(' ') for row in electronic.data_words_trigrams]
