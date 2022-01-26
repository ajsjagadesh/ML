import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

df = pd.read_csv('emails.csv')   # Loading CSV file
df = df.fillna(method ='pad')  

print(df.info())

# Dividing Data into (X indipendent veribels) & (Y dependent veribels)
X = df.drop(['Email No.','Prediction'], axis = 1)
Y = df['Prediction']


# Divideing Data for traing & testing 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# Create LogisticRegression() Model
clf = LogisticRegression()

# Traning model
clf.fit(X_train,Y_train)
s = clf.score(X_test,Y_test) # accurecy test
print(s)

#output 1 = spam, 0 = not spam
