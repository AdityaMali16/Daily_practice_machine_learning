import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

mapping = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10
}

df=pd.read_csv('hiring/hiring_data.csv')
df['experience']=df["experience"].str.lower().map(mapping)
df['experience']=df["experience"].fillna(0)
df['test_score(out of 10)']=df["test_score(out of 10)"].fillna(df["test_score(out of 10)"].median())
print(df.head())


new_df=df.drop('salary($)',axis='columns')
salary=df['salary($)']
model=linear_model.LinearRegression()
model.fit(new_df,salary)

print(model.predict([[2,9,6]]))
print(model.predict([[12,10,10]]))




