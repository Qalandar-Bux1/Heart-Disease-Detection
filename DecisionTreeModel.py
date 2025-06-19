import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('heart.csv')

data = load_data()

# Split features and target
X = data.drop('target', axis=1).values
y = data['target'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Train model
# Make sure your custom DecisionTree class is defined or imported correctly
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))

# ---- UI ----
st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Provide the patient's clinical parameters below to check the likelihood of heart disease.</p>", unsafe_allow_html=True)
st.divider()

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 29, 77, 54)
    sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.slider('Resting Blood Pressure (trestbps)', 90, 200, 120)
    chol = st.slider('Cholesterol Level (chol)', 100, 600, 240)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])

with col2:
    restecg = st.selectbox('Resting ECG Results (restecg)', [0, 1, 2])
    thalach = st.slider('Max Heart Rate Achieved (thalach)', 70, 210, 150)
    exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.selectbox('Slope of ST Segment (slope)', [0, 1, 2])
    ca = st.selectbox('Major Vessels Colored (ca)', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

# Predict button
st.markdown("### 🧮 Run Prediction")
if st.button("🔍 Predict Heart Disease"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    result = clf.predict(user_input)

    st.success("✅ No Heart Disease Detected!" if result[0] == 0 else "⚠️ Heart Disease Detected!")
    st.metric(label="🔎 Model Accuracy", value=f"{acc:.2%}")

# Footer
st.divider()
st.markdown(
    "<small style='text-align:center; display:block;'>Developed using a Custom Decision Tree Model | Dataset: UCI Heart Disease</small>",
    unsafe_allow_html=True
)
