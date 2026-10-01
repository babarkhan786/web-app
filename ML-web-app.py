# import libraries
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# title of app
st.write(""" 
         **Explore different models for datasets**
         """)
# dataset names in box
datasets_name = st.sidebar.selectbox('Select Dataset',
                                     ('Iris', 'Brest cancer', 'Wine')
)

# Classifiers names in box
classifier_name = st.sidebar.selectbox('Select Classifier',
                                       ('KNN', 'SVM', 'Random Forest')
)

# load datasets

def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y
# call   the function
X, y = get_dataset(datasets_name)

# shape of dataset
st.write('shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

# parameters of classifiers
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C',0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('max_depth', 2, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_septh', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

# call the function
params = add_parameter_ui(classifier_name)

# get classifiers names and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf

# call the clf function
clf = get_classifier(classifier_name, params)

# test the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# train classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# check the accuracy of the model
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

# plot PCA
pca = PCA(2)
X_projected = pca.fit_transform(X)
# dimension of plot
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x1,x2,
            c=y, alpha=0.8,
            cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal component 2')
plt.colorbar()
# show plot
st.pyplot(fig)