import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

flower=load_iris()

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train , X_test , y_train , y_test =train_test_split(flower.data,flower.target,test_size=0.3)

model=linear_model.LogisticRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)

for p in pred[:]:
    print(flower.target_names[p])

print(accuracy_score(y_test,pred))