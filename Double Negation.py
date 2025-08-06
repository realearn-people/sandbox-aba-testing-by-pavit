import pandas as pd
import nltk
from nltk.corpus import wordnet

# Check NLTK data necessay
def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading 'punkt' for tokenization...")
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading 'punkt_tab' for tokenization...")
        nltk.download('punkt_tab')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading 'averaged_perceptron_tagger' for POS tagging...")
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading 'averaged_perceptron_tagger_eng' for POS tagging...")
        nltk.download('averaged_perceptron_tagger_eng')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading 'wordnet' for antonym lookup...")
        nltk.download('wordnet')
    print("NLTK data is ready.")

def generate_double_negation(sentence: str) -> str:
    """
    Generates a double negation version of a sentence by replacing the first
    adjective found with a negated antonym from WordNet.
    
    Args:
        sentence: The input sentence string.

    Returns:
        A new sentence string with a double negation phrase, or the original
        sentence if no suitable adjective/antonym is found.
    """
    # Tokenize the sentence into words and apply Part-of-Speech tags
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)

    new_sentence_parts = list(tokens)
    # We use a flag to ensure we only replace one word per sentence for clarity and to avoid overly complex generated sentences.
    has_replaced = False

    for i, (word, tag) in enumerate(tagged_tokens):
        # We target adjectives (JJ: adjective, JJR: adj., comparative, JJS: adj., superlative)
        if tag.startswith('JJ') and not has_replaced:
            antonyms = []
            # Use WordNet to find antonyms for the adjective
            for syn in wordnet.synsets(word, pos=wordnet.ADJ):
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        # Add the name of the antonym's lemma to our list
                        antonyms.append(lemma.antonyms()[0].name())
            
            if antonyms:
                # Select the first found antonym for simplicity
                antonym = antonyms[0].replace('_', ' ')
                
                # Construct the double negation phrase, currently only 'not' is used
                negated_antonym_phrase = f"not {antonym}"
                
                # Replace the original word in our list of parts
                new_sentence_parts[i] = negated_antonym_phrase
                has_replaced = True
    
    # Reconstruct the sentence from the parts
    return ' '.join(new_sentence_parts).replace(" ,", ",").replace(" .", ".").replace(" 's", "'s")

# Main
if __name__ == "__main__":
    download_nltk_data()
    
    try:
        # Load the dataset from the provided CSV file
        df = pd.read_csv("Original ABA Dataset for Version 3 [May 30] - 1. hotel in Larnaca-Cyprus - Topic.csv")

        # Filter for rows where 'Sentiment' is 'Positive'
        positive_reviews = df[df['Pos/Neg'] == 'Positive'].copy()
        
        print(f"\n--- Found {len(positive_reviews)} positive reviews. ---")
        print("Applying double negation to the first sentence of a few examples:\n")
        
        generated_sentences = []
        for index, row in positive_reviews.head(25).iterrows():
            original_review = row['PositiveReview']
            # Split review into sentences and process the first one
            first_sentence = nltk.sent_tokenize(original_review)[0]
            generated_sentence = generate_double_negation(first_sentence)
            generated_sentences.append(generated_sentence)
            
            print(f"Original:  '{first_sentence}'")
            print(f"Generated: '{generated_sentence}'\n" + "-"*25)

    except FileNotFoundError:
        print("Error: The CSV file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")