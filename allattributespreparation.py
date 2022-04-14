#Get attributes
list_cat = ['Electronic Device', 'Food & Grocery', 'Clothes', 'Books', 'Home Appliance']
distribution = pd.DataFrame()
for cat in list_cat:
  distribution = pd.concat([distribution, pd.read_csv(f'/content/drive/My Drive/PhD/JOM_data/distribution_{cat}.csv').drop('Unnamed: 0', axis=1)])

distribution = distribution.reset_index(drop=True)

#Get label
list_cat_ = ['elec','food','clothes','book','home']
elec = pd.read_csv('/content/drive/My Drive/PhD/JOM_data/label_elec.csv').drop('Unnamed: 0', axis=1)
food = pd.read_csv('/content/drive/My Drive/PhD/JOM_data/label_food.csv').drop('Unnamed: 0', axis=1)
clothes = pd.read_csv('/content/drive/My Drive/PhD/JOM_data/label_clothes.csv').drop('Unnamed: 0', axis=1)
book = pd.read_csv('/content/drive/My Drive/PhD/JOM_data/label_book.csv').drop('Unnamed: 0', axis=1)
home = pd.read_csv('/content/drive/My Drive/PhD/JOM_data/label_home.csv').drop('Unnamed: 0', axis=1) 

label = pd.concat([elec, food, clothes, book, home])

label = label.reset_index(drop=True)

X = distribution.loc[:, distribution.columns != 'text'].join([label.iloc[:,1:]])

#Encode product category
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X.category = le.fit_transform(X.category.values.reshape(-1,1))
X.columns

#Information retrieval for each latent topic as supplement materials for labelling


#Get Dominant Topics for each review
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=electronic.text_cleaned_phase1.reset_index(drop=True)):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts={....}.text_cleaned_phase1.reset_index(drop=True))#replace with name of each product category

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(50)
df_dominant_topic.to_csv('/content/drive/My Drive/PhD/JOM_data/df_dominant_topic_{...}.csv') #replace with name of each product category

#Load reviews with dominant topic
list_ = ['elec','book','clothes','home','food']
full_df = pd.DataFrame()
for name in list_:
  cat_df = pd.read_csv(f'/content/drive/My Drive/PhD/JOM_data/df_dominant_topic_{name}.csv').drop(['Unnamed: 0'], axis=1)
  cat_df['Category'] = name
  full_df = pd.concat([full_df, cat_df]).reset_index(drop=True)
list_sents = full_df.Text.values

# create a df containing sentence and its output for rule 1
row_list = []

for sent in tqdm(list_sents):
    output = rule1(sent)
    dict1 = {'Output':output}
    row_list.append(dict1)
    
df_rule1 = pd.DataFrame(row_list)

# rule 1 achieves 20% result on simple sentences
output_per(df_rule1,'Output')

df_rule1['category'] = full_df['Category']
df_rule1['Topic'] = full_df['Dominant_Topic']

# selecting non-empty outputs
df_show1 = []

for row in tqdm(range(len(df_rule1))):
    cat = df_rule1.loc[row,'category']
    top = df_rule1.loc[row,'Topic']
    output = df_rule1.loc[row,'Output']
    if len(output)!=0:
      for sent in output:
        df_show1.append({'Output':sent, 'category':cat, 'topic':top})

# reset the index
df_show1 = pd.DataFrame(df_show1)
df_show1.reset_index(inplace=True)
df_show1.drop('index',axis=1,inplace=True)


# create a df containing sentence and its output for rule 2
row_list = []

for sent in tqdm(list_sents):
  
    # rule
    output = rule2(sent)
    dict1 = {'Output':output}
    row_list.append(dict1)

df_rule2 = pd.DataFrame(row_list)

df_rule2['category'] = full_df['Category']
df_rule2['Topic'] = full_df['Dominant_Topic']

# selecting non-empty outputs
df_show2 = []

for row in tqdm(range(len(df_rule2))):
    cat = df_rule2.loc[row,'category']
    top = df_rule2.loc[row,'Topic']
    output = df_rule2.loc[row,'Output']
    if len(output)!=0:
      for sent in output:
        df_show2.append({'Output':sent, 'category':cat, 'topic':top})

# reset the index
df_show2 = pd.DataFrame(df_show2)
df_show2.reset_index(inplace=True)
df_show2.drop('index',axis=1,inplace=True)

# create a df containing sentence and its output for rule 3
row_list = []

for i in tqdm(list_sents):
  # rule
  output = rule3(i)
  dict1 = {'Output':output}
  row_list.append(dict1)

df_rule3 = pd.DataFrame(row_list)

df_rule3['category'] = full_df['Category']
df_rule3['Topic'] = full_df['Dominant_Topic']

# selecting non-empty outputs
df_show3 = []

for row in tqdm(range(len(df_rule3))):
    cat = df_rule2.loc[row,'category']
    top = df_rule2.loc[row,'Topic']
    output = df_rule3.loc[row,'Output']
    if len(output)!=0:
      for sent in output:
        df_show3.append({'Output':sent, 'category':cat, 'topic':top})

# reset the index
df_show3 = pd.DataFrame(df_show3)
df_show3.reset_index(inplace=True)
df_show3.drop('index',axis=1,inplace=True)

#Join all topics

#Get representative variables
customer_service = pd.Series(X[['Clothes6']].mean(axis=1), name=('Customer Service'))
value_for_money = pd.Series(X[['Clothes8','Home1','Home14']].mean(axis=1), name=('Value for Money'))
#payment_invoice = X[[]].mean(axis=1)
ease_of_use = pd.Series(X[['Home13']].mean(axis=1),  name='Ease of Use')
performance = pd.Series(X[['Books3','Books8','Clothes13','F&G5','Elec1','Elec3','Elec7','Elec8','Elec14','Home3','Home9','Home16']].mean(axis=1),
                        name='Performance (Usage)')
appearance = pd.Series(X[['Books4','Clothes3','Clothes5','Clothes7','Clothes9','Clothes14','F&G1','F&G2','Elec10','Home11']].mean(axis=1), 
                       name='Physical Appearance')
durability = pd.Series(X[['Clothes2','Clothes12','F&G3','Elec5','Home2','Home5']].mean(axis=1), name='Durability')
reliability = pd.Series(X[['Books5','Clothes4','F&G4','Elec13','Home10','Home12']].mean(axis=1), name='Reliability')
accessories = pd.Series(X[['Clothes11','Elec2','Elec4','Elec11','Elec12']].mean(axis=1), name='Second Features')
design = pd.Series(X[['Books1','Books2','Books6','Books7','Clothes1','Clothes10','Home6','Home7','Home8','Home15']].mean(axis=1),
                   name='Design')
#brand = pd.Series(X[['Clothes6']].mean(axis=1), name='brand')
returns = pd.Series(X[['Elec9']].mean(axis=1), name='Return-related Operations')
packaging = pd.Series(X[['F&G6','Elec6','Home4']].mean(axis=1), name='Packaging')
#logistics = X[[]].mean(axis=1)
new_X = pd.concat([customer_service, value_for_money, ease_of_use,
                performance, appearance, durability, reliability, accessories,
                design, returns, packaging], axis=1)

new_X = new_X.join(X[['category','helpfulness_numerical', 'length', 'sentiment', 'cus_image_numerical']])

new_X.columns = ['Customer Service', 'Value for Money', 'Ease of Use',
       'Performance', 'Physical Appearance', 'Durability',
       'Reliability', 'Secondary Features', 'Primary Features', 'Return Management',
       'Packaging', 'Product Category','Review Helpfulness','Review Length','Review Sentiment','Review Image']

y = label.label
new_X['Product Category'] = new_X['Product Category'].astype('category').cat.as_ordered()
new_X['Customer Images'] = new_X['Review Image'].astype('category').cat.as_ordered()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3,
                                                    random_state = 5, stratify = y)
