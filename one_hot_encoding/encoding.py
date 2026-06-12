import pandas as pd
import numpy as np
from sklearn import linear_model


df=pd.read_csv('one_hot_encoding/carprices.csv')

# dummies=pd.get_dummies(df['Car Model'])

# df_dummies=pd.concat([df,dummies],axis='columns')

# df_dummies=df_dummies.drop(['Car Model','Audi A5'],axis='columns')


# price=df['Sell Price($)']
# df=df_dummies.drop(['Sell Price($)'],axis='columns')

# reg= linear_model.LinearRegression()
# reg.fit(df,price)

# print(reg.predict([[45000,4,0,1]]))
# print(reg.predict([[86000,7,1,0]]))
# print(reg.predict([[86000,7,1,0]]))

# print(reg.score(df,price))


#doing one hot encoding with the help of sklearn library
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
dfle=df
dfle['Car Model']=le.fit_transform(df['Car Model'])



X=dfle[['Car Model','Mileage','Age(yrs)']].values
y=dfle['Sell Price($)'].values

from sklearn.preprocessing import OneHotEncoder
cato=dfle['Car Model']
print(cato)
ohe=OneHotEncoder(categories=cato)
X_encoded=ohe.fit_transform(X).toarray()
# print(X_encoded)




