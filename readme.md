
# Step 1: Data Preparation
Load the data: Ensure the data is correctly loaded into a DataFrame.
Clean the data: Handle any missing values or duplicates. Clean the lyrics text if necessary (e.g., removing special characters).

# Step 2: Text Preprocessing with nltk
Tokenization: Split the lyrics into individual words or tokens.
Stop Words Removal: Remove common stop words (e.g., "the", "and", "a").
Lemmatization: Reduce words to their base or root form.

# Step 3: Exploratory Data Analysis (EDA)
Basic Statistics: Calculate basic statistics such as the number of albums, tracks, and total lyrics.
Word Count: Calculate the total word count for each song.
Unique Word Count: Calculate the number of unique words for each song.
Most Common Words: Identify the most common words in the entire dataset.
Most Common Words per Album
Visuals: Word Frequency Distributions and Word Clouds.

# Step 4:Text Analysis
Sentiment Analysis: Analyze the sentiment of each song's lyrics (e.g., positive, negative, neutral).
N-gram Analysis: Analyze the most common bi-grams or tri-grams (pairs or triplets of words). --> PHRASES
TF-IDF Analysis: Compute the Term Frequency-Inverse Document Frequency to identify important words.

# Step 5: Advanced Analysis 
Topic Modeling: Use techniques like LDA (Latent Dirichlet Allocation) to identify topics within the lyrics. --> MEANINGS
Text Clustering: Cluster songs based on their lyrical content.
Sentiment Over Time: Analyze how the sentiment of the lyrics changes across different albums or years.

# Notes 13/6/2023

Need to rethink/reevaluate the text preprocessing:
1) stop words 
2) abbreviations like i'll --> ill, you've -->youve
3) exclamations like woohoo? or nana?



