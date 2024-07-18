
# Step 1: Data Preparation
1. Load the data: Ensure the data is correctly loaded into a DataFrame.
2. Clean the data: Handle any missing values or duplicates. Clean the lyrics text if necessary (e.g., removing special characters).

# Step 2: Text Preprocessing with nltk
1. Tokenization: Split the lyrics into individual words or tokens.
2. Stop Words Removal: Remove common stop words (e.g., "the", "and", "a").
3. Lemmatization: Reduce words to their base or root form.

# Step 3: Exploratory Data Analysis (EDA)
1. Basic Statistics: Calculate basic statistics such as the number of albums, tracks, and total lyrics.
2. Word Count: Calculate the total word count for each song.
3. Unique Word Count: Calculate the number of unique words for each song.
4. Most Common Words: Identify the most common words in the entire dataset.
5. Most Common Words per Album
6. Visuals: Word Frequency Distributions and Word Clouds.

# Step 4:Text Analysis
1. Sentiment Analysis: Analyze the sentiment of each song's lyrics (e.g., positive, negative, neutral).
2. N-gram Analysis: Analyze the most common bi-grams or tri-grams (pairs or triplets of words). --> PHRASES
3. TF-IDF Analysis: Compute the Term Frequency-Inverse Document Frequency to identify important words (read tf_idf.md file)

# Step 5: Advanced Analysis 
1. Topic Modeling: Use techniques like LDA (Latent Dirichlet Allocation) to identify topics within the lyrics. --> MEANINGS
2. Text Clustering: Cluster songs based on their lyrical content.
3. Sentiment Over Time: Analyze how the sentiment of the lyrics changes across different albums or years.
4. Linguistic complexitiy. Readability library.
6. Use Spotify Database or api
https://developer.spotify.com/documentation/web-api/reference/get-audio-features


# Notes 
## 13/6/2024

Need to rethink/reevaluate the text preprocessing:
1) stop words X words_to_exclude, additional_stopwords
2) abbreviations like i'll --> ill, you've -->youve X
3) exclamations like woohoo? or nana? X words_to_exclude, additional_stopwords

## 21/6/2024
4) What to do with ngrams? Evolution? 
5) Should I use tokens insted of cleaned lyrics?

## 24/06/2024
6) TF-IDF holds major potential in identifying most important words. 
7) analysis per song (document)? or per album (set of documents)?
8) Visualize the importance of some words using time graphs (tf_idf score vs song_id).

## 28/06/2024

9. Text Classification vs Sentiment Analysis are different approaches.
10. Design different preprocessing and normalization for each.
11. Figure out normalization methods and what to look for in each case.
12. Can I use text analysis to figure out important words and then use aspect SA?

https://arxiv.org/pdf/1703.00607v2 useful paper?

## 29/06/2024

13. Performed LBSA and SA using pre-trained distilBERT. Notebook analysis on sentiment.ipynb

## 07/05/2024 toDo:

14. Test 10 models and create classification metrics.
15. Determine thresholds.
16. Set up evaluation metrics.

##
### Descriptive Statistics

- Calculate and compare:
  - Mean confidence score, pdf positive
  - Median
  - Standard deviation for sentiment scores from each model.

### Visualization positive pdf

- Histograms: Frequency distribution of sentiment scores.
- Box Plots: Quartiles comparison and outlier detection.
- Density Plots: Estimate and visualize the probability density function.

### Visualize transformer_labels
### Visualize transformer_confidence

### Statistical Tests

- Kolmogorov-Smirnov test to compare distributions.


# Next Steps

Sentiment was a good rabbit hole BUT, 

1. LDA to identify ESG topics
2. ABSA for these topics to see how it evolves in discography. 
3. Keep album sentiment as a baseline and draw connections with vailance. 