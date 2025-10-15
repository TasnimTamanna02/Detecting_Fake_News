# Detecting_Fake_News using PassiveAggressiveClassification

In this notebook, I build a text classification model to automatically distinguish between fake and real news articles. The model uses TF-IDF vectorization and a Passive-Aggressive Classifier for fast, scalable performance suitable for real-time systems. Although this project focuses on news content, the same approach can be extended to other domains such as spam detection, rumor classification, or misinformation monitoring.

#Problem Statement
With the exponential growth of online media, misinformation has become a major global issue. The goal of this project is to design a machine learning model that can classify news articles as FAKE or REAL based on their textual content. Traditional detection systems rely on manual review or rule-based filters, which are limited and slow. Using machine learning and NLP, we can automatically learn language patterns that indicate deceptive or factual reporting.

#Dataset
The news.zip contains the dataset (news.csv file) used to train and test this model. It has 6,335 news article records.
Each record includes:
<li> title – headline of the news article </li>
<li>text – full content of the article</li>
<li>label – target variable (FAKE or REAL)</li>
The dataset is relatively balanced, ensuring fair model training and testing.

#Techniques Used
**Text Preprocessing**
To ensure cleaner and more meaningful text representation, the following preprocessing steps were applied:
<li>Lowercasing all text</li>
<li>Removing punctuation, stopwords, URLs, HTML tags, and extra spaces</li>
<li>Lemmatization using NLTK (reducing words to their base form)</li>
<li>Creating a TF-IDF matrix to convert text into numerical features</li>

#Model Used: Passive-Aggressive Classifier
This algorithm is efficient for large-scale text classification tasks such as spam filtering or news verification. It works well for online learning, making it adaptable for continuous updates from streaming data.
**Reasons to use Passive-Aggressive Classifier:**
<li>Designed for real-time classification</li>
<li>High accuracy with minimal resource use</li>
<li>Robust against outliers and noise in text data</li>
<li>Ideal for binary classification problems like this one</li>

#Model Training and Evaluation
After preprocessing, the dataset was split into 80% training and 20% testing subsets.
A TF-IDF vectorizer was used to generate feature vectors for both sets.

**Performance Metrics:**
<li>Accuracy</li>
<li>Precision</li>
<li>Recall</li>
<li>F1-score</li>

**Confusion Matrix**
