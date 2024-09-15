library(readxl)
library(dplyr)
library(ggplot2)

# Disable scientific notation for the current session
options(scipen = 999)


filePath <- "data/merged_coldplay_lyrics_sentiment_audio_features.xlsx"

data <- read_excel(filePath)

data |> glimpse()



# Assuming your data is already loaded into 'data'
filtered_data <- data %>%
  select(`Track Name`, `Track Number`, `Album Name`, `Album Release Date`, sentiment_final,
         danceability, energy, key, loudness, mode, speechiness, acousticness, 
         instrumentalness, liveness, valence, tempo, duration_ms, time_signature)

# Preview the filtered data
glimpse(filtered_data)

# Numeric columns
  numeric_columns <- c("danceability", "energy", "loudness", "speechiness", "acousticness", 
                       "instrumentalness", "liveness", "valence", "tempo", "duration_ms")
  
  # Categorical-like columns
  categorical_columns <- c("key", "mode", "time_signature")
  

# Function to create plots for numeric and categorical features
plot_audio_features <- function(df) {
  
  # Numeric columns
  numeric_columns <- c("danceability", "energy", "loudness", "speechiness", "acousticness", 
                       "instrumentalness", "liveness", "valence", "tempo", "duration_ms")
  
  # Categorical-like columns
  categorical_columns <- c("key", "mode", "time_signature")
  
  # Plot numeric features using histograms and density plots
  for (col in numeric_columns) {
    p <- ggplot(df, aes(x = .data[[col]])) +
      geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "blue", alpha = 0.6) +
      geom_density(color = "red") +
      ggtitle(paste("Distribution of", col)) +
      theme_minimal()  
    print(p)
  }

}


# Function to create a General Distribution Plot for a given continuous feature
plot_general_distribution <- function(df, feature_name) {
  ggplot(df, aes(x = .data[[feature_name]])) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "blue", alpha = 0.6) +
    geom_density(color = "red", size = 1) +
    ggtitle(paste("General Distribution of", feature_name)) +
    xlab(feature_name) +
    ylab("Density") +
    theme_minimal()
}

# Example usage for 'danceability'
plot_general_distribution(filtered_data, "danceability")


# Function to create a General Distribution Plot for a given continuous feature
# with a custom figure size of (12, 8)
plot_general_distribution <- function(df, feature_name, feature_min = NULL, feature_max = NULL) {
  
  # Adjust bin width for features with small ranges
  if (!is.null(feature_min) & !is.null(feature_max)) {
    bin_width <- (feature_max - feature_min) / 30  # Adjust bins for smaller ranges
  } else {
    bin_width <- 0.05  # Default bin width for normal ranges
  }
  
  p <- ggplot(df, aes(x = .data[[feature_name]])) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, binwidth = bin_width, fill = "blue", alpha = 0.6) +
    geom_density(color = "red", size = 1) +
    ggtitle(paste("General Distribution of", feature_name)) +
    xlab(feature_name) +
    ylab("Density") +
    theme_bw()
  
  # Display the plot and save it with custom figure size
  print(p)
  ggsave(filename = paste0("plot_", feature_name, ".png"), plot = p, width = 12, height = 8)
}



# Function to create Distribution Per Album in Chronological Order
plot_distribution_per_album <- function(df, feature_name, theme_choice = "light") {
  
  # Select theme based on user input
  theme_selected <- switch(theme_choice,
                           "light" = theme_light(),
                           "minimal" = theme_minimal(),
                           "classic" = theme_classic(),
                           "bw" = theme_bw(),
                           theme_light())  # Default is theme_light
  
  p <- ggplot(df, aes(x = reorder(`Album Name`, `Album Release Date`), y = .data[[feature_name]])) +
    geom_boxplot(fill = "blue", alpha = 0.6) +
    ggtitle(paste("Distribution of", feature_name, "per Album in Chronological Order")) +
    xlab("Album Name") +
    ylab(feature_name) +
    theme_selected +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Display the plot and save it with custom figure size
  print(p)
  ggsave(filename = paste0("plot_", feature_name, "_by_album.png"), plot = p, width = 12, height = 8)
}





# Function to create Distribution by Sentiment and Album in Chronological Order
plot_feature_by_sentiment_by_album <- function(df, feature_name, theme_choice = "light") {
  
  # Filter the data to include only POSITIVE and NEGATIVE sentiments
  df_filtered <- df %>%
    filter(sentiment_final %in% c("POSITIVE", "NEGATIVE"))
  
  # Select theme based on user input
  theme_selected <- switch(theme_choice,
                           "light" = theme_light(),
                           "minimal" = theme_minimal(),
                           "classic" = theme_classic(),
                           "bw" = theme_bw(),
                           theme_light())  # Default is theme_light
  
  p <- ggplot(df_filtered, aes(x = sentiment_final, y = .data[[feature_name]], fill = sentiment_final)) +
    geom_boxplot(alpha = 0.6) +
    facet_wrap(~ reorder(`Album Name`, `Album Release Date`), scales = "free") +  # Facet by Album in chronological order
    ggtitle(paste("Distribution of", feature_name, "by Sentiment and Album")) +
    xlab("Sentiment (POSITIVE vs NEGATIVE)") +
    ylab(feature_name) +
    scale_fill_manual(values = c("POSITIVE" = "green", "NEGATIVE" = "red")) +
    theme_selected +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate axis labels for better readability
  
  # Display the plot and save it with custom figure size
  print(p)
  ggsave(filename = paste0("plot_", feature_name, "_by_sentiment_by_album.png"), plot = p, width = 12, height = 8)
}


# Function to create a line plot showing Feature Trends Over Time by Album
plot_feature_trends_over_time <- function(df, feature_name, theme_choice = "light") {
  
  # Select theme based on user input
  theme_selected <- switch(theme_choice,
                           "light" = theme_light(),
                           "minimal" = theme_minimal(),
                           "classic" = theme_classic(),
                           "bw" = theme_bw(),
                           theme_light())  # Default is theme_light
  
  p <- ggplot(df, aes(x = `Album Release Date`, y = .data[[feature_name]], color = `Album Name`)) +
    geom_line(aes(group = `Album Name`), size = 1) +
    geom_point(size = 3) +
    ggtitle(paste("Trend of", feature_name, "Over Time")) +
    xlab("Album Release Date") +
    ylab(feature_name) +
    theme_selected
  
  # Display the plot and save it with custom figure size
  print(p)
  ggsave(filename = paste0("plot_", feature_name, "_trends_over_time.png"), plot = p, width = 12, height = 8)
}

library(ggplot2)

# Function to create a bar plot of Top Tracks by Feature in Descending Order
plot_top_tracks_by_feature <- function(df, feature_name, top_n = 10, theme_choice = "light") {
  
  # Sort the data by the feature and select the top N tracks
  top_tracks <- df %>%
    arrange(desc(.data[[feature_name]])) %>%
    slice(1:top_n)  # Get top N tracks
  
  # Select theme based on user input
  theme_selected <- switch(theme_choice,
                           "light" = theme_light(),
                           "minimal" = theme_minimal(),
                           "classic" = theme_classic(),
                           "bw" = theme_bw(),
                           theme_light())  # Default is theme_light
  
  p <- ggplot(top_tracks, aes(x = reorder(`Track Name`, .data[[feature_name]]), y = .data[[feature_name]], fill = `Album Name`)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    coord_flip() +  # Flip for better readability
    ggtitle(paste("Top", top_n, "Tracks by", feature_name)) +
    xlab("Track Name") +
    ylab(feature_name) +
    theme_selected
  
  # Display the plot and save it with custom figure size
  print(p)
  ggsave(filename = paste0("plot_top_", top_n, "_tracks_", feature_name, ".png"), plot = p, width = 12, height = 8)
}

# Example usage for 'danceability'
plot_top_tracks_by_feature(filtered_data, "danceability", top_n = 10, theme_choice = "light")

# Example usage for 'energy'
plot_top_tracks_by_feature(filtered_data, "energy", top_n = 10, theme_choice = "minimal")




#1 Example usage for 'danceability' (range 0 to 1)
plot_general_distribution(filtered_data, "danceability", feature_min = 0, feature_max = 1)

#2 Example usage for 'danceability'
plot_distribution_per_album(filtered_data, "danceability", theme_choice = "light")

#3 Example usage for 'energy'
plot_feature_by_sentiment_by_album(filtered_data, "energy", theme_choice = "minimal")

#4 Example usage for 'danceability'
plot_feature_trends_over_time(filtered_data, "danceability", theme_choice = "light")

#5 Example usage for 'danceability'
plot_top_tracks_by_feature(filtered_data, "danceability", top_n = 10, theme_choice = "light")




# Function to create a bar plot of Tracks in Descending Order by Feature for a Specific Album
# with Top 3 Tracks in a Different Color
plot_tracks_by_feature_for_album <- function(df, feature_name, album_name, theme_choice = "light") {
  
  # Filter the data to only include tracks from the specified album
  album_tracks <- df %>%
    filter(`Album Name` == album_name) %>%
    arrange(desc(.data[[feature_name]])) %>%
    mutate(rank = row_number())  # Rank the tracks by the selected feature
  
  # Select theme based on user input
  theme_selected <- switch(theme_choice,
                           "light" = theme_light(),
                           "minimal" = theme_minimal(),
                           "classic" = theme_classic(),
                           "bw" = theme_bw(),
                           theme_light())  # Default is theme_light
  
  # Create the plot, highlighting the top 3 tracks in a different color
  p <- ggplot(album_tracks, aes(x = reorder(`Track Name`, .data[[feature_name]]), y = .data[[feature_name]], fill = rank <= 3)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    scale_fill_manual(values = c("TRUE" = "red", "FALSE" = "blue"), guide = "none") +  # Red for top 3, blue for the rest
    coord_flip() +  # Flip for better readability
    ggtitle(paste("Tracks from", album_name, "by", feature_name, "(Descending Order)")) +
    xlab("Track Name") +
    ylab(feature_name) +
    theme_selected
  
  # Display the plot and save it with custom figure size
  print(p)
  ggsave(filename = paste0("plot_tracks_", feature_name, "_", album_name, "_top3.png"), plot = p, width = 12, height = 8)
}

# Example usage for 'danceability' in album 'Parachutes'
plot_tracks_by_feature_for_album(filtered_data, "danceability", "Parachutes", theme_choice = "light")

# Updated Numeric columns (excluding tempo and duration_ms)
numeric_columns <- c("danceability", "energy", "loudness", "speechiness", "acousticness", 
                     "instrumentalness", "liveness", "valence")

# Loop through numeric columns and produce all graphs
for (feature in numeric_columns) {
  
  # General Distribution Plot (range 0 to 1 for all features)
  plot_general_distribution(filtered_data, feature, feature_min = 0, feature_max = 1)
  
  # Distribution Per Album in Chronological Order
  plot_distribution_per_album(filtered_data, feature, theme_choice = "light")
  
  # Feature by Sentiment and Album in Chronological Order
  plot_feature_by_sentiment_by_album(filtered_data, feature, theme_choice = "light")
  
  # Feature Trends Over Time
  plot_feature_trends_over_time(filtered_data, feature, theme_choice = "light")
  
  # Top Tracks by Feature
  plot_top_tracks_by_feature(filtered_data, feature, top_n = 10, theme_choice = "light")
}



# Get a unique list of album names from the dataset
album_names <- unique(filtered_data$`Album Name`)

# Loop through each album and each numeric column to produce the plots
for (album in album_names) {
  for (feature in numeric_columns) {
    # Create the plot for each album and numeric feature
    plot_tracks_by_feature_for_album(filtered_data, feature, album, theme_choice = "light")
  }
}