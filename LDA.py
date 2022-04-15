%%time
import gensim
from gensim.models.wrappers import LdaMallet
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim import similarities
from gensim.models import LsiModel
from tqdm import tqdm
import re
import glob
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    """
    Compute c_v coherence for various number of topicså

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LSI topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values_cv = []


    for num_topics in tqdm(range(start, limit, step)):
        lsi_model = LdaMallet(mallet_path,
                      corpus=corpus,
                      id2word=id2word,
                      num_topics=num_topics,
                      random_seed = 1,
                      iterations= 1000,
                      optimize_interval = 10
                      )

        coherencemodel_cv = CoherenceModel(model=lsi_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values_cv.append(coherencemodel_cv.get_coherence())

    return coherence_values_cv

def prepare_data(dataset):
  data_lemmatized = [row.split(' ') for row in dataset.data_words_trigrams]
  # Create Dictionary
  id2word = corpora.Dictionary(data_lemmatized)
  id2word.filter_extremes(no_below=0.01, no_above = 0.99)
  # Create Corpus
  texts = data_lemmatized

  # Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in texts]
  return id2word, corpus, data_lemmatized
#save all tuning results from topic models
from pandas import ExcelWriter
# from pandas.io.parsers import ExcelWriter
def save_xls(list_dfs, xls_path, cohe_names):
    with ExcelWriter(xls_path) as writer:
        for df, name in zip(list_dfs, cohe_names):
            df.to_excel(writer,'{}'.format(name))
        writer.save()
# Can take a long time to run.
cats = [food, book, fashion, elec, home]
names = ['Food & Grocery', 'Book','Fashion Item','Electronic Device','Home Appliance']

for cat, cat_name in tqdm(zip(cats, names)):
  id2word, corpus, data_lemmatized = prepare_data(cat)
  coherence_values_cv = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=52, step=1)
  list_dfs = [pd.DataFrame(coherence_values_cv,columns=['cv'])]
  save_xls(list_dfs, f'/content/drive/My Drive/PhD/JOM_data/{cat_name}_LDAMALLET.xls'.format(cat_name), cohe_names)

# Get the number of topics with the highest coherence score
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
limit=52; start=2; step=1;
x = range(start, limit, step)
print('The optimal number of topics: ' + str(pd.DataFrame(list(zip(x, coherence_values_cv))).set_index(0).idxmax(axis=0).iloc[0]))

# Plot the results
fig = plt.figure(figsize=(15, 7))

plt.plot(
    x,
    coherence_values_cv,
    linewidth=3,
    color='#4287f5'
)

plt.xlabel("Numbers of Topics", fontsize=25)
plt.ylabel("Coherence Score", fontsize=25)

plt.xticks(np.arange(2, max(x) + 1, 2), fontsize=20)
plt.yticks(fontsize=20)

plt.show()

# Get topic distribution from LDA MALLET

import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized}) #customise data_lemmatized for each product category
id2word.filter_extremes(no_below=0.01, no_above = 0.99
                        )
# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

%%time
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim import similarities


import re
import glob
#                    random_seed=1,
#                    alpha='auto',
#                    iterations=1000,
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# Build LDA MALLET model

lda_mallet = LdaMallet(mallet_path,
                      corpus=corpus,
                      id2word=id2word,
                      num_topics= ...., #'optimal number of topics for each product category'
                      random_seed = 1,
                      iterations= 1000,
                      optimize_interval = 10,
                      alpha = 1
                      )
lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_mallet)
from gensim.models.coherencemodel import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
from pprint import pprint

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics(num_topics = ...., num_words=30))  #'optimal number of topics for each product category'
doc_lda = lda_model[corpus]

#Get distribution of topics per document
cat = 'Name of product category'

distribution = []

for i, row in enumerate(lda_model[corpus]):
  distribution.append([per for per in row])
distribution = [dict(doc) for doc in distribution]
def change_name(k=0):
  dict_ = {}
  for i in range(k):
    dict_[i] = f'Name of product category{i+1}'
  return dict_
distribution = pd.DataFrame(distribution).rename(columns = change_name(...))#'optimal number of topics for each product category'
distribution = distribution.fillna(0)
distribution['category'] = cat
distribution.to_csv(f'/content/drive/My Drive/PhD/JOM_data/distribution_{cat}.csv')
