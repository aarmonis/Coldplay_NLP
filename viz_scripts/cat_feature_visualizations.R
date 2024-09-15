library(ggplot2)
library(dplyr)
library(readxl)
# Disable scientific notation for the current session
options(scipen = 999)

# Load the Excel file (adjust the path as necessary)
file_path <- "data/merged_coldplay_lyrics_sentiment_audio_features.xlsx"  # Adjust path accordingly
filtered_data <- read_excel(file_path)

# Categorical columns
categorical_columns <- c("key", "mode", "time_signature")

# Binned columns
binned_columns <- c("tempo", "duration_ms")

### Function to create count plot for a categorical feature by album with light theme
plot_categorical_feature_per_album <- function(df, feature_name, feature_label) {
  
  # Create count plot
  p <- ggplot(df, aes(x = reorder(`Album Name`, `Album Release Date`), fill = factor(.data[[feature_name]]))) +
    geom_bar(position = "fill") +
    ggtitle(paste("Distribution of", feature_label, "by Album")) +
    xlab("Album") +
    ylab("Proportion") +
    theme_light() +  # Apply light theme
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = guide_legend(title = feature_label)) +
    scale_fill_viridis_d(name = feature_label)  # Optional: Using viridis color scale
  
  print(p)
  # Optionally save the plot
  ggsave(filename = paste0("plot_", feature_name, "_by_album.png"), plot = p, width = 12, height = 8)
}

### Function to create binned bar plot for a feature by album with light theme
plot_binned_feature_per_album <- function(df, feature_name, breaks, feature_label) {
  
  df <- df %>%
    mutate(binned_feature = cut(.data[[feature_name]], breaks = breaks))
  
  # Create binned bar plot
  p <- ggplot(df, aes(x = binned_feature)) +
    geom_bar(fill = "blue", alpha = 0.7) +
    facet_wrap(~ reorder(`Album Name`, `Album Release Date`)) +  # Facet by album
    ggtitle(paste("Distribution of", feature_label, "in Bins by Album")) +
    xlab(feature_label) +
    theme_light() +  # Apply light theme
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p)
  ggsave(filename = paste0("plot_", feature_name, "_binned_by_album.png"), plot = p, width = 12, height = 8)
}

### Loop through categorical columns to generate visualizations with light theme
for (feature in categorical_columns) {
  
  # Generate count plots for each categorical feature
  if (feature == "key") {
    plot_categorical_feature_per_album(filtered_data, feature, feature_label = "Musical Key")
  } else if (feature == "mode") {
    plot_categorical_feature_per_album(filtered_data, feature, feature_label = "Mode")
  } else if (feature == "time_signature") {
    plot_categorical_feature_per_album(filtered_data, feature, feature_label = "Time Signature")
  }
}

### Loop through binned columns (tempo, duration_ms) to generate visualizations with light theme
for (feature in binned_columns) {
  
  # Define breaks for each feature
  if (feature == "tempo") {
    breaks <- seq(0, 300, by = 20)  # Custom breaks for tempo
    plot_binned_feature_per_album(filtered_data, feature, breaks, feature_label = "Tempo (BPM)")
    
  } else if (feature == "duration_ms") {
    breaks <- seq(0, 600000, by = 60000)  # Custom breaks for duration
    plot_binned_feature_per_album(filtered_data, feature, breaks, feature_label = "Duration (ms)")
  }
}
