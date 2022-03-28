import pandas as pd
import numpy as np
import tensorflow as tf


dataset = pd.read_csv("agaricus-lepiota.data", sep=",", header=None)
#print(dataset.info())

y = dataset.iloc[:,0].values
X = dataset.iloc[:,1:]

#Dropping the 11th variable because it is the only variable that has missing values which is 2480
X = X.drop(X.columns[11], axis = 1)


#   PreProcessing

#Encoding X values by Frequency Map
X_values = {}
for i in X.columns:
    for j in X.columns:
        X_values = X[i].value_counts().to_dict()
        X[i] = X[i].map(X_values)
        X_values.clear()

X = X.values

#Label Encoding for the y values (1 for poisoned / 0 for edible)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Scaling X values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#   Building the ANN
ann = tf.keras.models.Sequential()
#Adding the input layer and the first layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
#Adding the second layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

#   Training the ANN
#Compiling the ANN
ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"] )
#Training the ANN on Training Set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Predicting the Test Set Resulst
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
le_yp = LabelEncoder()
y_pred = le_yp.fit_transform(y_pred)


#Accuracy 
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


















