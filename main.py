import numpy as np 
import pandas as pd 
import matplotlib.pyplot as p
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

data = pd.read_csv('C:/Users/Vaishnu/Desktop/LSTM_SentimentAnalysis/demonetization-tweets.csv',encoding='ISO-8859-1')

from nltk.sentiment import vader
from nltk.sentiment.util import *

from nltk import tokenize
sid = vader.SentimentIntensityAnalyzer()
data['sentiment_compound_polarity']=data.text.apply(lambda x:sid.polarity_scores(x)['compound'])

data['sentiment_negative']=data.text.apply(lambda x:sid.polarity_scores(x)['neg'])
data['sentiment_pos']=data.text.apply(lambda x:sid.polarity_scores(x)['pos'])
data['sentiment_neu']=data.text.apply(lambda x:sid.polarity_scores(x)['neu'])
data['sentiment']=''
data.loc[data.sentiment_compound_polarity>0,'sentiment']=1
data.loc[data.sentiment_compound_polarity==0,'sentiment']=0
data.loc[data.sentiment_compound_polarity<0,'sentiment']=-1
data.head()
 
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size)

positive_count, negative_count, neutral_count=0, 0, 0
for x in range(len(X_train)):
    if Y_train[x][1] == 1:
        negative_count += 1
    elif Y_train[x][0] == 1:
        positive_count += 1
    elif Y_train[x][2] == 1:
        neutral_count +=1  
        
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

pos_cnt, neg_cnt, pos_correct, neg_correct, neu_cnt, neu_correct = 0, 0, 0, 0, 0, 0
for x in range(len(X_test)):
    
    result = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_test[x]):
        if Y_test[x][1] == 1:
            neg_correct += 1
        elif Y_test[x][0] == 1:
            pos_correct += 1
        elif Y_test[x][2] == 1:
            neu_correct +=1  
       
    if Y_test[x][1] == 1:
        neg_cnt += 1
    elif Y_test[x][0] == 1:
        pos_cnt += 1
    elif Y_test[x][2] == 1:
        neu_cnt += 1

print("neg_cnt", neg_cnt)
print("pos_cnt", pos_cnt)
print("neu_cnt", neu_cnt)
"""
print("neg_corect", neg_correct)
print("pos_correct", pos_correct)
print("neu_correct", neu_correct)
"""

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height, '%d' % int(height), ha='center', va='bottom')


x_title=["Positive","Negative","Neutral"]

val=data['sentiment'].value_counts().reset_index()
#val=[pos_correct, neg_correct, neu_correct]
val.columns=['Sentiment','Count']
fig, ax = p.subplots()
rect1=ax.bar(x_title, val.Count)
autolabel(rect1)
p.title('SA with LSTM(Complete Dataset Result)')
p.show()

val1=[neu_cnt, neg_cnt, pos_cnt]
fig, ax = p.subplots()
rect2=ax.bar(x_title, val1, color = 'red')
autolabel(rect2)
p.title('SA with LSTM(Test set)')
p.xlabel('Tweets')
p.ylabel('Tweet Count')
p.show()

val2=[neutral_count, negative_count, positive_count]
fig, ax = p.subplots()
rect3=ax.bar(x_title, val2, color = 'green')
autolabel(rect3)
p.title('SA with LSTM(Train set)')
p.xlabel('Tweets')
p.ylabel('Tweet Count')
p.show()

def largest(arr,n): 
    max = arr[0] 
    for i in range(1, n): 
        if arr[i] > max: 
            max = arr[i] 
    return max

def TestTweet(twt):
    twt = tokenizer.texts_to_sequences(twt)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=146, dtype='int32', value=0)
    print(twt)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
    max=largest(sentiment,len(sentiment))
    if(sentiment[1] == max):
        print("Neutral")
    elif (sentiment[2] == max):
        print("Positive")
    elif (sentiment[0] == max):
        print("Negative")

twt = [data['text'][3]]
TestTweet(twt)
