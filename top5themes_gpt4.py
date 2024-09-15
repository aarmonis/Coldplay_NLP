import openai
import pandas as pd

# Set up your OpenAI API key
from openai import OpenAI

client = OpenAI(api_key=
'',
)

# Function to interact with GPT-4.5 Turbo and extract themes and topics
def analyze_lyrics_with_gpt(lyrics, task="Summarize the themes and topics"):
    prompt = f"Lyrics:\n{lyrics}\n\nTask: {task}"
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts themes and topics from song lyrics."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Function to get the top 5 themes and topics for each album
def summarize_album_topics(df):
    albums = df['Album Name'].unique()  # Get unique albums
    album_summaries = {}
    
    for album in albums:
        album_data = df[df['Album Name'] == album]
        combined_lyrics = " ".join(album_data['lyrics_clean'])  # Combine all lyrics for the album
        
        # Analyze themes and topics for the album using GPT-4.5 Turbo
        print(f"Processing album: {album}...")
        themes_summary = analyze_lyrics_with_gpt(combined_lyrics, task="Summarize the top 5 themes and topics in this album")
        
        album_summaries[album] = themes_summary  # Store the result for each album
        print(f"Themes for {album}: {themes_summary}\n")
    
    return album_summaries

# Function to save the summaries to a text file
def save_summaries_to_file(summaries, filename='album_themes_summary.txt'):
    with open(filename, 'w') as f:
        for album, summary in summaries.items():
            f.write(f"Album: {album}\n")
            f.write(f"Themes and Topics:\n{summary}\n")
            f.write("\n" + "-"*80 + "\n")

# Load your dataset (adjust the path as necessary)
file_path = "data/merged_coldplay_lyrics_sentiment_audio_features.xlsx"  # Replace with actual path
df = pd.read_excel(file_path)

# Ensure the lyrics are preprocessed and clean (this assumes a column named 'lyrics_clean')
df['lyrics_clean'] = df['lyrics_clean'].fillna('')  # Handle any missing lyrics

# Summarize the themes and topics for each album
album_summaries = summarize_album_topics(df)

# Save the summaries to a file
save_summaries_to_file(album_summaries, filename='album_themes_summary.txt')

# Print final output
print("Summaries have been saved to 'album_themes_summary.txt'.")
