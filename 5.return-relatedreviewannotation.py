#get return label 
text = data_return_verified.text_cleaned_phase1
index_numb = data_return_verified.index.values
#returns keywords from amazon website
returns_vocab_level1a = ['return'] 
returns_vocab_level1b = ['\\brma\\b','\\bra\\b','\\brga\\b']
returns_vocab_level2 = ['send[A-Za-z0-9\s+]{0,30}back','sent[A-Za-z0-9\s+]{0,30}back',
                        'ship[A-Za-z0-9\s+]{0,30}back','post[A-Za-z0-9\s+]{0,30}back','money[A-Za-z0-9\s+]{0,30}back'
                        ]
#keyword of refund and money
returns_vocab_level3 = ['refund','reimburse','repay','repaid','warranty','repair','replace','replacing','guarantee','money back'
                        ]

#filter out all reviews that have 'return' vocab level 1a
return_label_1a = []
index_return_label_1a = []
remaining_reviews1a = []
index_numb_remaining_1a = []


for ind_num, line in tqdm(zip(index_numb, text)):
    regex = re.compile("|".join(returns_vocab_level1a))
    if re.search(regex, line):

        return_label_1a.append(line)
        index_return_label_1a.append(ind_num)
    else: 
        remaining_reviews1a.append(line)
        index_numb_remaining_1a.append(ind_num)


#filter out all reviews that have 'return' vocab level 1b
return_label_1b = []
index_return_label_1b = []
remaining_reviews1b = []
index_numb_remaining_1b = []


for ind_num, line in tqdm(zip(index_numb_remaining_1a,remaining_reviews1a)):
    regex = re.compile("|".join(returns_vocab_level1b))
    if re.search(regex, line):
        return_label_1b.append(line)
        index_return_label_1b.append(ind_num)

    else: 
        remaining_reviews1b.append(line)
        index_numb_remaining_1b.append(ind_num)
        
#filter out all reviews that have 'return' vocab level 2
#create a regex that can capture the pattern of sen+....+back or ship+...+back
return_label_2 = []
index_return_label_2 = []
remaining_reviews2 = []
index_numb_remaining_2 = []


for ind_num, line in tqdm(zip(index_numb_remaining_1b,remaining_reviews1b)):
    pattern = re.compile(r"send[A-Za-z0-9\s+]{0,30}back|sent[A-Za-z0-9\s+]{0,30}back|ship[A-Za-z0-9\s+]{0,30}back|post[A-Za-z0-9\s+]{0,30}back")
    if re.search(pattern, line):

        return_label_2.append(line)
        index_return_label_2.append(ind_num)
    else: 
        remaining_reviews2.append(line)
        index_numb_remaining_2.append(ind_num)


#filter out all reviews that have 'return' vocab level 3
return_label_3 = []
index_return_label_3 = []
remaining_reviews3 = []
index_numb_remaining_3 = []


for ind_num, line in tqdm(zip(index_numb_remaining_2 ,remaining_reviews2)):
    regex = re.compile("|".join(returns_vocab_level3))
    if re.search(regex, line):

        return_label_3.append(line)
        index_return_label_3.append(ind_num)
    else: 
        remaining_reviews3.append(line)
        index_numb_remaining_3.append(ind_num)


""" Classification model """
joined_index = index_return_label_1a + index_return_label_1b + index_return_label_2 + index_return_label_3
joined_index = pd.DataFrame(joined_index)
joined_index['label'] = 1
joined_index = joined_index.set_index(0)

data_return_verified['label'] = joined_index
data_return_verified = data_return_verified.fillna(0)

