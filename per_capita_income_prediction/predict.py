import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


df=pd.read_csv("canada_per_capita_income.csv")
print(df.head())


plt.scatter(df["year"],df["per capita income (US$)"] , color="r" , marker='+')
plt.xlabel("year")
plt.ylabel("per capita income")
plt.show()

new_df=df.drop('per capita income (US$)' , axis='columns')
per_capita_income=df['per capita income (US$)']

reg=linear_model.LinearRegression()
reg.fit(new_df , per_capita_income)

print(reg.predict([[2020]]))