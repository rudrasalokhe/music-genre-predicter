import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn import tree

musicData = pd.read_csv('music.csv')
x = musicData.drop(columns=['genre'])
y = musicData['genre']
model = DecisionTreeClassifier()
model.fit(x, y)
predictions = model.predict([[21, 1]])
predictions
