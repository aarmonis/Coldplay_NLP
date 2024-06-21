
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
3. TF-IDF Analysis: Compute the Term Frequency-Inverse Document Frequency to identify important words.

# Step 5: Advanced Analysis 
1. Topic Modeling: Use techniques like LDA (Latent Dirichlet Allocation) to identify topics within the lyrics. --> MEANINGS
2. Text Clustering: Cluster songs based on their lyrical content.
3. Sentiment Over Time: Analyze how the sentiment of the lyrics changes across different albums or years.

# Notes 13/6/2023

Need to rethink/reevaluate the text preprocessing:
1) stop words 
2) abbreviations like i'll --> ill, you've -->youve
3) exclamations like woohoo? or nana?
4) 



