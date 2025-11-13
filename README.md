# Disaster Tweet Classifier

An NLP pipeline that automatically categorizes disaster-related tweets to identify urgent needs (food, shelter, medical aid) and prioritize emergency response efforts.

## Overview

This project analyzes 10,000+ disaster-related tweets using machine learning to classify urgent needs during emergencies. The system achieved **83% accuracy** with a **25% precision improvement** over baseline keyword matching, enabling faster resource allocation during disaster response.

## Technologies

- **Python** - Core implementation
- **scikit-learn** - Logistic Regression classifier
- **Pandas & NumPy** - Data processing and numerical operations
- **NLTK** - Text preprocessing and tokenization

## Pipeline

**1. Data Preprocessing**
- Text cleaning (removal of URLs, punctuation, special characters)
- Tokenization and normalization
- Stop word removal

**2. Feature Engineering**
- **Bag-of-Words (CountVectorizer)**: Converts text to numerical feature vectors
- **Custom Lexical Features**: Domain-specific keywords for urgency detection (e.g., "urgent", "help", "need")
- Feature extraction optimized for disaster-related terminology

**3. Classification**
- Logistic Regression for multi-class classification
- Categories: food, shelter, medical aid, other
- Train-test split with stratified sampling

**4. Evaluation**
- Confusion matrix analysis
- Precision, recall, and F1-score per category
- 83% overall classification accuracy

## Key Results

✓ **83% accuracy** across all need categories  
✓ **25% precision improvement** over rule-based baseline  
✓ **10,000+ tweets** processed and classified  
✓ Real-time classification capability for emergency response

## Project Structure

```
DisasterRelief_Section1.ipynb  # Rule-based classifier baseline
DisasterRelief_Section2.ipynb  # ML model with CountVectorizer
DisasterRelief_Section3.ipynb  # Word embeddings (GloVe) exploration
```

## Usage Example

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Train vectorizer and model
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_tweets)
model = LogisticRegression()
model.fit(train_vectors, train_labels)

# Classify new tweet
new_tweet = ["Urgent need medical supplies shelter 5th Street"]
new_vector = vectorizer.transform(new_tweet)
prediction = model.predict(new_vector)
```

## Impact

This classifier enables emergency response teams to:
- Automatically triage social media messages during disasters
- Identify urgent resource needs in real-time
- Allocate aid more efficiently based on detected patterns
- Reduce response time for critical situations

---

*Developed as part of NLP coursework demonstrating machine learning applications in humanitarian response.*
