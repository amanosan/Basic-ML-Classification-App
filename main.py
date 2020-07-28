import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.title("Basic Machine Learning App.")

st.write("""
## Exploring different Classifiers.
""")

dataset_name = st.sidebar.selectbox("Select a Dataset",
("Iris Dataset", "Breast Cancer Dataset", "Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select a Classifier",
("KNN", "SVM", "Random Forest"))

# Getting our dataset
def get_dataset(dataset_name):
    if dataset_name == 'Iris Dataset':
        df = datasets.load_iris()
    elif dataset_name == "Breast Cancer Dataset":
        df = datasets.load_breast_cancer()
    else:
        df = datasets.load_wine()
    X = df.data
    y = df.target

    return X, y

# Getting our classfier model
def get_classfier(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    elif classifier_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    max_depth=params["max_depth"], random_state=1234)
    
    return clf

# To give a UI for selecting parameters
def add_parameter_ui(classifier_name):
    params = dict()

    if classifier_name == 'KNN':
        n_neighbors = st.sidebar.slider("K-value", 1, 15)
        params["n_neighbors"] = n_neighbors
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C-value", 0.01, 10.0)
        params["C"] = C
    else: 
        max_depth = st.sidebar.slider("Max-Depth", 2, 15)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params

# getting our data
X, y = get_dataset(dataset_name)

# information about our data
st.write(f"Shape of Datset: {X.shape}")
st.write(f"Number of Classes: {len(np.unique(y))}")

# parameters for our model
params = add_parameter_ui(classifier_name)

# creating our model
model = get_classfier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write(f"Classifier : {classifier_name}")
st.write(f"Accuracy : {accuracy * 100:.2f}%")

# Plotting 
pca = PCA(2)

X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()
# plt.show() 
st.pyplot()