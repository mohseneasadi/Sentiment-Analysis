# Capstone Project : Sentiment Analysis program

# Import Modules
import numpy as np
import pandas as pd
import spacy 
from textblob import TextBlob
from spacytextblob.spacytextblob import SpacyTextBlob

# Load NLP Pipeline
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# load the data 
amazon_df = pd.read_csv('amazon_product_reviews.csv')

# remove all missing values
reviews_data = amazon_df[['reviews.text']].dropna()

# Creat function to preprocess the text data
def preprocess_text(text):

    doc = nlp(text.lower().strip())

    # Tokenize the text
    processed_text = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    
    # Join the tokens back into a string
    return " ".join(processed_text)

reviews_data['processed.text'] = reviews_data['reviews.text'].apply(preprocess_text)

# sentiment analysis for the amazon_product_reviews dataset
data = reviews_data['processed.text']

# Create a function for sentiment analysis
def setiment_analysis(text):
    
    doc = nlp(text.lower().strip())
    
    sentiment = doc._.blob.subjectivity
    sentiment = round(sentiment,2)
    polarity  = doc._.blob.polarity
    polarity  = round(polarity,2)

# checking the polarity     
    if polarity > 0:
        review_setiment = "Positive"
    elif polarity < 0:
        review_setiment = "Negative"
    else : 
        review_setiment = "Neutral"
    
    print(f'\nReview_label : {review_setiment}\n \nSentiment Analysis: \nPolarity: {polarity} subjectivity: {sentiment}')


# running some examples
print("-----------------------------------------------------------------")
print(f'\nReview: {data[4]}')
test = setiment_analysis(data[4])

print("-----------------------------------------------------------------")
print(f'\nReview: {data[450]}')
test = setiment_analysis(data[450])
print("-----------------------------------------------------------------")
