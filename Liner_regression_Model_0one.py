import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
import joblib
import pickle


# Loading CSV file
df = pd.read_csv("Salary_Data.csv") 

# ploting data 
%matplotlib inline
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(df.YearsExperience,df.Salary)
#plt.plot(df.YearsExperience,df.Salary)

# Dividing Data into (X indipendent veribels) & (Y dependent veribels)
X = df.drop('Salary',axis='columns')
Y = df['Salary']


# Divideing Data for traing & testing 
X_tr, X_te, Y_tr, Y_te = model_selection.train_test_split(X,Y, test_size=0.20, random_state=0)

# Create linear regression Model
model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)

# Traning model
model.fit(X_tr, Y_tr)


v=model.predict(X_te) # predicting Model
print(X_te)
print(v)
print()
print(model.score(X_te,Y_te)) # accurecy test

print(model.coef_)         #finding coeficent
print(model.intercept_)    #finding Interception ( Y = M*x + B)   
                           # M = coeficent
                           # B = Interception & X = value wiche we want to predict



# Ploting the prediction line
%matplotlib inline
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(df.YearsExperience,df.Salary, color='red')
plt.plot(df.YearsExperience, model.predict(df[['YearsExperience']]),color='blue') 

#saving model using pickle
with open('model_pickle', 'wb') as f:
  pickle.dump(model,f)

#Loading Model using pickle
with open('model_pickle', 'rb') as f:
  model0 = pickle.load(f) 

model0.predict(X_te)



"""
#saving Model using joblib
joblib.dump(model, 'model_joblib')

#Loading Model using joblib
md = joblib.load('model_joblib')
md.predict(X_te)

"""
