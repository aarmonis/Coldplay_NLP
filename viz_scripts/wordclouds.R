# Install necessary libraries if not already installed
# install.packages("wordcloud")
# install.packages("tm")

library(ggplot2)
library(dplyr)
library(readxl)
library(wordcloud)
library(tm)

# Load the dataset (adjust the path as necessary)
file_path <- "data/merged_coldplay_lyrics_sentiment_audio_features.xlsx"  # Adjust the path accordingly
filtered_data <- read_excel(file_path)

# Function to preprocess lyrics and generate word clouds for each album
plot_wordcloud_by_album <- function(df) {
  
  # Get the unique list of albums
  albums <- unique(df$`Album Name`)
  
  for (album in albums) {
    
    # Filter lyrics for the current album
    album_data <- df %>%
      filter(`Album Name` == album) %>%
      pull(lyrics_clean) %>%
      paste(collapse = " ")  # Collapse all lyrics into a single text for the album
    
    # Create a corpus from the lyrics
    corpus <- Corpus(VectorSource(album_data))
    
    # Text preprocessing: Convert to lower case, remove punctuation, numbers, and stop words
    corpus <- tm_map(corpus, content_transformer(tolower))
    corpus <- tm_map(corpus, removePunctuation)
    corpus <- tm_map(corpus, removeNumbers)
    corpus <- tm_map(corpus, removeWords, stopwords("en"))
    
    # Create a term-document matrix
    tdm <- TermDocumentMatrix(corpus)
    tdm_matrix <- as.matrix(tdm)
    
    # Calculate word frequencies
    word_freqs <- sort(rowSums(tdm_matrix), decreasing = TRUE)
    word_freq_df <- data.frame(word = names(word_freqs), freq = word_freqs)
    
    # Generate the word cloud for the current album
    png(filename = paste0("wordcloud_", gsub(" ", "_", album), ".png"), width = 800, height = 800)
    wordcloud(words = word_freq_df$word, freq = word_freq_df$freq, min.freq = 1,
              max.words = 100, random.order = FALSE, rot.per = 0.35,
              colors = brewer.pal(8, "Dark2"))
    dev.off()  # Save the word cloud as PNG file
  }
}

### Generate Word Clouds for Each Album ###

plot_wordcloud_by_album(filtered_data)
