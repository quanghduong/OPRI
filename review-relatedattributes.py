#extract helpfulness
def helpfulness(x):
  if x == 0:
    return 0
  elif re.search('One', x):
    return 1
  else:
    return int(re.match(r'\d+', x)[0])
data_return_verified['helpfulness_numerical'] = data_return_verified['helpfulness'].apply(lambda x: helpfulness(x))

#extract length of reviews
def length(x):
  x = x.split(' ')
  return len(x)
data_return_verified['length'] = data_return_verified['text_cleaned_phase1'].apply(lambda x: length(x))

#extract sentiment
from textblob import TextBlob
def sentiment(x):
  blob = TextBlob(x)
  return blob.sentiment.polarity
data_return_verified['sentiment'] = data_return_verified['text_cleaned_phase1'].apply(sentiment)

#extract customer image:
def cus_image(x):
  if len(str(x)) > 1:
    return 1
  else:
    return 0
data_return_verified['cus_image_numerical'] = data_return_verified['cus_image'].apply(cus_image)

#save in five different datasets
book =  data_return_verified.iloc[:,26:][data_return_verified.key == 'Books']
clothes = data_return_verified.iloc[:,26:][data_return_verified.key == 'Clothes']
food_grocery = data_return_verified.iloc[:,26:][data_return_verified.key == 'Food & Grocery']
elec =  data_return_verified.iloc[:,26:][data_return_verified.key == 'Electronic Device']
home = data_return_verified.iloc[:,26:][data_return_verified.key == 'Home Appliance']
book.to_csv('/content/drive/My Drive/PhD/JOM_data/label_book.csv')
clothes.to_csv('/content/drive/My Drive/PhD/JOM_data/label_clothes.csv')
food_grocery.to_csv('/content/drive/My Drive/PhD/JOM_data/label_food.csv')
elec.to_csv('/content/drive/My Drive/PhD/JOM_data/label_elec.csv')
home.to_csv('/content/drive/My Drive/PhD/JOM_data/label_home.csv')

