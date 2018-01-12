#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:21:30 2018

@author: virajdeshwal
"""

'''Let's start the Natural Language Processing.
    In the dataset.
    1= positive review
    0= negetive review'''

import pandas as pd

#We will use tab seperated Variabel(.tsv) file.
#checkout the syntax for .tsv below
'''quoting =3 becasue we are eliminating the double quotes'''
file = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting=3)

#cleaning the text
'''We will get rid of all the 
    STOP WORDS - unneccessary words 
    and of punctuations.'''
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
corpus=[]

#stop = input('Press any  key to remove stop words...\n')
nltk.download('stopwords')
print('\nDone. You did a great job\n')
#stem = input('Press any key to do the stemming of the words....\n')


for i in range(0,1000):
    review = re.sub('[^a-zA-z]',' ', file['Review'][i])
    review = review.lower()


    review = review.split()

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]


    #joining back the split word to make the string

    review = ' '.join(review)
    corpus.append(review)


print('\n\nWhoa you did a great job!\n')
print('\n We are done with the preprocessing step of Restaurant review.\nNow Lets  create a ML model for restaurant review\n')

#ml = input ('Press any key to create Bag of words model.\n')
#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
#independent variable
X = cv.fit_transform(corpus).toarray()
#dependent variable 

y = file.iloc[:,1].values
''' We can use any classification model to train the data now.
As we already have the X and y . Use any classification model from ML.'''
print('\nWOOOHOOO you did a great job!! MODEL IS READY!!!\n')
'''I am using Naive Bayes Classifier'''



from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=0)



'''Call the model library. You are free to check the accuracy with any of the below algo.
Just remove the comment and use the algo. I am using NB because it is best for NLP.'''
from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cluster import KMeans
#from sklearn.svm import SVC
#initialize the model

model = GaussianNB()
model.fit(x_train, y_train)
#predict the model
y_pred = model.predict(x_test)

#import the confusion matrix
from sklearn.metrics import confusion_matrix

#show the true positive and false positive through the confusion matrix.
conf_matrix = confusion_matrix(y_test, y_pred)
print('\n\n print the confusion matrix for true and false prediction rate.\n\n')
print(conf_matrix)

plt.imshow(conf_matrix)
plt.title('Confustion Matrix for the TP TN FP FN')

plt.show()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('\n\n\n Hence the accuracy of the GaussianNB is:',accuracy)
print('\nCongratulations you predicted with good accuracy on this small set of datasets. CHEERS!!\n')

