import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:/Users/BhavyaSri/Desktop/Iris.csv')
dataset.head()
dataset.describe()
dataset["Species"].unique()
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
print(classifier.predict([[5.6,2.8,4.9,2.0]]))
from sklearn.metrics import accuracy_score
print("Accuracy score",accuracy_score(y_test, y_pred)*100,"%")
import gradio as gr
def flower(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
    global classifier
    species = classifier.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    return species[0]

demo = gr.Interface(fn=flower,description="Flower Spacies \n\n Please enter the value between 1 to 10", inputs=[gr.inputs.Number(label="SepalLengthCm"), gr.inputs.Number(label="SepalWidthCm"),gr.inputs.Number(label="PetalLengthCm"), gr.inputs.Number(label="PetalWidthCm")], outputs="text")

demo.launch(share=True)