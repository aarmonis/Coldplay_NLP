library(ggplot2)
library(dplyr)
library(readxl)

# Load the dataset (adjust the path as necessary)
file_path <- "data/merged_coldplay_lyrics_sentiment_audio_features.xlsx"  # Adjust the path accordingly
filtered_data <- read_excel(file_path)

# Function to visualize the overall sentiment distribution (Pie Chart with Light Theme and Percentage Labels)
plot_sentiment_distribution <- function(df) {
  sentiment_counts <- df %>%
    count(sentiment_final) %>%
    mutate(proportion = n / sum(n) * 100,  # Calculate percentage
           label = paste0(round(proportion, 1), "%"))  # Format label with percentage
  
  p <- ggplot(sentiment_counts, aes(x = "", y = proportion, fill = sentiment_final)) +
    geom_bar(stat = "identity", width = 1) +
    coord_polar("y", start = 0) +
    ggtitle("Sentiment Distribution (POSITIVE vs NEGATIVE)") +
    scale_fill_manual(values = c("POSITIVE" = "green", "NEGATIVE" = "red")) +
    theme_light() +  # Apply the light theme
    theme(axis.title.x = element_blank(),  # Hide x-axis title
          axis.title.y = element_blank(),  # Hide y-axis title
          axis.text = element_blank(),     # Hide axis text
          axis.ticks = element_blank(),    # Hide axis ticks
          panel.border = element_blank(),  # Remove panel border
          panel.grid = element_blank(),    # Remove grid lines
          legend.title = element_blank())  # Remove legend title
  
  # Add the percentage labels on the pie chart
  p <- p + geom_text(aes(label = label), position = position_stack(vjust = 0.5), size = 5)
  
  print(p)
  ggsave("plot_sentiment_distribution_pie.png", plot = p, width = 8, height = 8)
}



# Function to visualize sentiment distribution by album (Stacked Bar Plot)
plot_sentiment_by_album <- function(df) {
  p <- ggplot(df, aes(x = reorder(`Album Name`, `Album Release Date`), fill = sentiment_final)) +
    geom_bar(position = "fill") +
    ggtitle("Sentiment Distribution by Album") +
    xlab("Album") +
    ylab("Proportion") +
    scale_fill_manual(values = c("POSITIVE" = "green", "NEGATIVE" = "red")) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    xlab("Album (Chronological Order)")
  print(p)
  ggsave("plot_sentiment_by_album.png", plot = p, width = 12, height = 8)
}

# Function to track sentiment trends over time (Line Plot)
plot_sentiment_trends_over_time <- function(df) {
  sentiment_trend <- df %>%
    group_by(`Album Release Date`, sentiment_final) %>%
    summarise(count = n()) %>%
    mutate(proportion = count / sum(count))
  
  p <- ggplot(sentiment_trend, aes(x = `Album Release Date`, y = proportion, color = sentiment_final, group = sentiment_final)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    ggtitle("Sentiment Trends Over Time") +
    xlab("Album Release Date") +
    ylab("Proportion") +
    scale_color_manual(values = c("POSITIVE" = "green", "NEGATIVE" = "red")) +
    theme_light()
  print(p)
  ggsave("plot_sentiment_trends_over_time.png", plot = p, width = 12, height = 8)
}

### Generate Sentiment Visualizations ###

# 1. Overall Sentiment Distribution (Pie Chart)
plot_sentiment_distribution(filtered_data)

# 2. Sentiment Distribution by Album (Chronological Order)
plot_sentiment_by_album(filtered_data)

# 3. Sentiment Trends Over Time
plot_sentiment_trends_over_time(filtered_data)
