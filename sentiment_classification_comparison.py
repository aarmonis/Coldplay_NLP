import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('coldplay_lyrics.csv')

# Preprocess the text data
df['lyrics_clean'] = df['lyrics'].str.lower().str.replace('[^\w\s]', '')

# Initialize sentiment analysis pipelines for different models
models = {
    'siebert/sentiment-roberta-large-english': pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english'),
    'cardiffnlp/twitter-roberta-base-sentiment': pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment'),
    'electra-base-discriminator-finetuned-imdb': pipeline('sentiment-analysis', model='jialicheng/electra-base-imdb'),
    'textattack/albert-base-v2-SST-2': pipeline('sentiment-analysis', model='textattack/albert-base-v2-SST-2'),
    'xlnet-base-cased-sentiment': pipeline('sentiment-analysis', model='dipawidia/xlnet-base-cased-product-review-sentiment-analysis'),
    'nlptown/bert-base-multilingual-uncased-sentiment': pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment'),
    'distilbert-base-uncased-finetuned-sst-2-english': pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
}

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Function to get VADER sentiment
def get_vader_sentiment(text):
    return vader.polarity_scores(text)['compound']

# Function to get Hugging Face transformer sentiment
def get_transformer_sentiment(model_pipeline, text):
    result = model_pipeline(text)[0]
    return result['label'], result['score']

# Apply each model to the dataset and store results
for model_name, model_pipeline in models.items():
    df[f'{model_name}_sentiment'], df[f'{model_name}_score'] = zip(*df['lyrics_clean'].apply(lambda x: get_transformer_sentiment(model_pipeline, x)))

# Apply VADER to the dataset
df['vader_sentiment'] = df['lyrics_clean'].apply(get_vader_sentiment)

# Analyze and compare the results
plt.figure(figsize=(14, 7))
for model_name in models.keys():
    plt.hist(df[f'{model_name}_score'], bins=20, alpha=0.5, label=model_name)
plt.hist(df['vader_sentiment'], bins=20, alpha=0.5, label='vader')
plt.legend(loc='upper right')
plt.title('Sentiment Score Distributions')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Optional: Export the results to a CSV for further analysis
df.to_csv('huggingface_sentiment_analysis_results.csv', index=False)
