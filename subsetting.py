import pandas as pd

input_file_path = "./dataset/filtered_english_tweets.csv"
output_file_path = "./dataset/random_sample_1000.csv"

df = pd.read_csv(input_file_path)

df_sample = df.iloc[1:].sample(n=1000, random_state=42)  # Exclude first row, set random_state for reproducibility

df_sample = pd.concat([df.iloc[[0]], df_sample])

df_sample.to_csv(output_file_path, index=False, header=True)

