#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import nltk
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import opinion_lexicon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('opinion_lexicon')


# In[3]:


# Sample polarity words
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())


# In[4]:


# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# In[5]:


documents = []
with open('Reviews.csv', newline='',encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        documents.append(row[0])


# In[6]:


documents


# In[7]:


# Label the documents based on presence of polarity words
labels = []
for doc in documents:
    preprocessed_doc = preprocess_text(doc)
    if any(word in preprocessed_doc for word in positive_words):
        labels.append('positive')
    elif any(word in preprocessed_doc for word in negative_words):
        labels.append('negative')
    else:
        labels.append('neutral')  # If neither positive nor negative words are found


# In[12]:


labels


# In[8]:


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the documents into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(documents)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)


# In[9]:


# Initialize and train the Support Vector Machine classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = classifier.predict(X_test)


# In[10]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[11]:


# Example prediction
new_document = "I had bad experience"
preprocessed_new_doc = preprocess_text(new_document)
new_doc_tfidf = vectorizer.transform([preprocessed_new_doc])
prediction = classifier.predict(new_doc_tfidf)
print("Predicted sentiment for the document:", prediction[0])


# In[13]:


# Example prediction
new_document = "I had good breakfast in my home"
preprocessed_new_doc = preprocess_text(new_document)
new_doc_tfidf = vectorizer.transform([preprocessed_new_doc])
prediction = classifier.predict(new_doc_tfidf)
print("Predicted sentiment for the document:", prediction[0])

