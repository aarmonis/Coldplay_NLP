import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
try:
    df = pd.read_excel('Coldplay Research Project_Data.xlsx')
    logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise

# Preprocess the text data
df['lyrics_clean'] = df['Lyrics'].str.lower().str.replace(r'[^\w\s]', '')

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
    try:
        result = model_pipeline(text)[0]
        return result['label'], result['score']
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return "ERROR", 0.0

# Function to calculate average sentiment score
def calculate_average_sentiment(df, model_name):
    return df[f'{model_name}_normalized_score'].mean()

# Function to normalize sentiment scores
def normalize_sentiment_score(score, model_name):
    if 'roberta' in model_name.lower():
        return (score - 1) / 4  # RoBERTa models typically output 0-4
    elif 'albert' in model_name.lower() or 'electra' in model_name.lower() or 'bert' in model_name.lower():
        return score  # These models typically output 0-1
    elif 'xlnet' in model_name.lower():
        return (score * 2) - 1  # XLNet typically outputs 0-1, convert to -1 to 1
    else:
        return score  # Default case

# Apply each model to the dataset and store results
for model_name, model_pipeline in tqdm(models.items(), desc="Analyzing sentiments"):
    df[f'{model_name}_sentiment'], df[f'{model_name}_score'] = zip(*df['lyrics_clean'].apply(lambda x: get_transformer_sentiment(model_pipeline, x)))
    df[f'{model_name}_normalized_score'] = df[f'{model_name}_score'].apply(lambda x: normalize_sentiment_score(x, model_name))
    avg_sentiment = calculate_average_sentiment(df, model_name)
    logging.info(f"Average sentiment for {model_name}: {avg_sentiment:.4f}")

# Apply VADER to the dataset
df['vader_sentiment'] = df['lyrics_clean'].apply(get_vader_sentiment)
vader_avg = df['vader_sentiment'].mean()
logging.info(f"Average VADER sentiment: {vader_avg:.4f}")

# Analyze and compare the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))

# Plot for Hugging Face models
for model_name in models.keys():
    ax1.hist(df[f'{model_name}_normalized_score'], bins=20, alpha=0.5, label=model_name)
ax1.legend(loc='upper right')
ax1.set_title('Hugging Face Models Sentiment Score Distributions')
ax1.set_xlabel('Normalized Sentiment Score')
ax1.set_ylabel('Frequency')

# Plot for VADER
ax2.hist(df['vader_sentiment'], bins=20, alpha=0.5, label='VADER')
ax2.legend(loc='upper right')
ax2.set_title('VADER Sentiment Score Distribution')
ax2.set_xlabel('Sentiment Score')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Optional: Export the results to a CSV for further analysis
df.to_csv('sentiment_analysis_results.csv', index=False)
