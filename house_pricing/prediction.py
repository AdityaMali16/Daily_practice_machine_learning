import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
import joblib
df = pd.read_csv("house_pricing/homeprices.csv")
print(df.head())

plt.scatter(df['area'],df['price'],color='b', marker='*')
plt.xlabel('area')
plt.ylabel('price')
plt.show()

new_df=df.drop('price',axis='columns')
price=df['price']


# with open('house_pricing/house_pricing_pickle','rb') as f:
#     reg=pickle.load(f)

# reg=linear_model.LinearRegression()
# reg.fit(new_df,price)

# joblib.dump(reg,'modle_joblib')
reg=joblib.load('house_pricing/modle_joblib')
print(reg.predict([[3300]]))

df1=pd.read_csv('house_pricing/areas.csv')

df1['price']=reg.predict(df1)
print(df1.head())

# with open('house_pricing_pickle','wb') as file:
#     pickle.dump(reg,file) 