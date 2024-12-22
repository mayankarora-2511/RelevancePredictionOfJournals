import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import string
df = pd.read_csv("./data.csv")
stop = set(stopwords.words('english'))
punctuations = string.punctuation

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in punctuations])
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop])
    return text

df['clean_paragraph'] = df['corpus'].apply(preprocess)
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
X = vectorizer.fit_transform(df['clean_paragraph']).toarray()
y = df['Relevance']  # Ensure that labels are numerical or encoded
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gnb = GaussianNB()
gnb.fit(X_train , y_train)
y_pred = gnb.predict(X_test)
acc1 = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
y_test_labels = le.inverse_transform(np.argmax(y_test, axis=1))
y_pred_labels = le.inverse_transform(y_pred)
crgnb = classification_report(y_test_labels, y_pred_labels,output_dict=True)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cmgnb = confusion_matrix(y_test_labels, y_pred_labels)
print("\nConfusion Matrix:\n", cmgnb)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
cm_display = ConfusionMatrixDisplay(confusion_matrix = cmgnb)
cm_display.plot()
plt.title("Gaussian naive bayes")
plt.show()