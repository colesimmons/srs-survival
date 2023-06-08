"""
This script fetches embeddings for all unique flashcard front and back text
by calling OpenAI's API. By default, these are 1536-dimensional.

It saves the embeddings to a JSON file that maps from text -> embedding.
"""


import json
import openai
import os
import pandas as pd
from tqdm import tqdm


openai.api_key = ""
filename = "embeddings.json"

def main(input_filename = "review_history_with_text.csv"):
    # Get list of all unique flashcard front or back text
    df = pd.read_csv(input_filename)
    all_text = df.groupby('card_id')['front'].first().tolist() + df.groupby('card_id')['back'].first().tolist()

    # If the output file already exists, load the existing embeddings.
    # Then, only set to fetch embeddings for text that we don't already have.
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            embds = json.load(f)
            to_fetch = [t for t in to_fetch if t not in embds]
    else:
        embds = {}
        to_fetch = all_text

    if not len(to_fetch):
        return

    batch_size = 100
    for i in tqdm(range(0, len(to_fetch) // batch_size + 1)):
        batch = to_fetch[i * batch_size: (i + 1) * batch_size]
        response = openai.Embedding.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        for text, embedding in zip(batch, response['data']):
            embds[text] = embedding['embedding']

    # Save the dictionary as a JSON file
    with open(filename, 'w') as f:
        json.dump(embds, f)

main()