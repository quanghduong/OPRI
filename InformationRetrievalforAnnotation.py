nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# function to check output percentage for a rule
def output_per(df,out_col):
    
    result = 0
    
    for out in df[out_col]:
        if len(out)!=0:
            result+=1
    
    per = result/len(df)
    per *= 100
    
    return per
# function for rule 1: noun(subject), verb, noun(object)
def rule1(text):
    
    doc = nlp(text)
    
    sent = []
    
    for token in doc:
        
        # if the token is a verb
        if (token.pos_=='VERB'):
            
            phrase =''
            
            # only extract noun or pronoun subjects
            for sub_tok in token.lefts:
                
                if (sub_tok.dep_ in ['nsubj','nsubjpass']) and (sub_tok.pos_ in ['NOUN','PROPN','PRON']):
                    
                    # add subject to the phrase
                    phrase += sub_tok.text

                    # save the root of the verb in phrase
                    phrase += ' '+token.lemma_ 

                    # check for noun or pronoun direct objects
                    for sub_tok in token.rights:
                        
                        # save the object in the phrase
                        if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN','PROPN']):
                                    
                            phrase += ' '+sub_tok.text
                            sent.append(phrase)
            
    return sent

# create a df containing sentence and its output for rule 1
row_list = []

for sent in tqdm(....): # input list of return reviews
    output = rule1(sent)
    dict1 = {'Output':output}
    row_list.append(dict1)
    
df_rule1 = pd.DataFrame(row_list)

# rule 1 achieves 20% result on simple sentences
output_per(df_rule1,'Output')

# selecting non-empty outputs
df_show1 = []

for row in tqdm(range(len(df_rule1))):
    output = df_rule1.loc[row,'Output']
    if len(output)!=0:
      for sent in output:
        df_show1.append({'Output':sent})

# reset the index
df_show1 = pd.DataFrame(df_show1)
df_show1.reset_index(inplace=True)
df_show1.drop('index',axis=1,inplace=True)

# function for rule 2
def rule2(text):
    
    doc = nlp(text)

    pat = []
    
    # iterate over tokens
    for token in doc:
        phrase = ''
        # if the word is a subject noun or an object noun
        if (token.pos_ == 'NOUN')\
            and (token.dep_ in ['dobj','pobj','nsubj','nsubjpass']):
            
            # iterate over the children nodes
            for subtoken in token.children:
                # if word is an adjective or has a compound dependency
                if (subtoken.pos_ == 'ADJ') or (subtoken.dep_ == 'compound'):
                    phrase += subtoken.text + ' '
                    
            if len(phrase)!=0:
                phrase += token.text
             
        if  len(phrase)!=0:
            pat.append(phrase)
        
    
    return pat
# create a df containing sentence and its output for rule 2
row_list = []

for sent in tqdm(...): # input list of return reviews
  
    # rule
    output = rule2(sent)
    dict1 = {'Output':output}
    row_list.append(dict1)

df_rule2 = pd.DataFrame(row_list)

# selecting non-empty outputs
df_show2 = []

for row in tqdm(range(len(df_rule2))):

    output = df_rule2.loc[row,'Output']
    if len(output)!=0:
      for sent in output:
        df_show2.append({'Output':sent})

# reset the index
df_show2 = pd.DataFrame(df_show2)
df_show2.reset_index(inplace=True)
df_show2.drop('index',axis=1,inplace=True)

# rule 3 function
def rule3(text):
    
    doc = nlp(text)
    
    sent = []
    
    for token in doc:

        # look for prepositions
        if token.pos_=='ADP':

            phrase = ''
            
            # if its head word is a noun
            if token.head.pos_=='NOUN':
                
                # append noun and preposition to phrase
                phrase += token.head.text
                phrase += ' '+token.text

                # check the nodes to the right of the preposition
                for right_tok in token.rights:
                    # append if it is a noun or proper noun
                    if (right_tok.pos_ in ['NOUN','PROPN']):
                        phrase += ' '+right_tok.text
                
                if len(phrase)>2:
                    sent.append(phrase)
                
    return sent
# create a df containing sentence and its output for rule 3
row_list = []

for i in tqdm(....): # input list of return reviews
  # rule
  output = rule3(i)
  dict1 = {'Output':output}
  row_list.append(dict1)

df_rule3 = pd.DataFrame(row_list)

# selecting non-empty outputs
df_show3 = []

for row in tqdm(range(len(df_rule3))):

    output = df_rule3.loc[row,'Output']
    if len(output)!=0:
      for sent in output:
        df_show3.append({'Output':sent})

# reset the index
df_show3 = pd.DataFrame(df_show3)
df_show3.reset_index(inplace=True)
df_show3.drop('index',axis=1,inplace=True)
