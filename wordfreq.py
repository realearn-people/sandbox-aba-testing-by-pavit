import pandas as pd
from collections import Counter
import re

df = pd.read_csv('Original ABA Dataset for Version 3 [May 30] - 1. hotel in Larnaca-Cyprus - Topic.csv')

# Get unique reviews
positive_reviews = df['PositiveReview'].dropna().unique()
negative_reviews = df['NegativeReview'].dropna().unique()

def simple_term_frequency(reviews, top_n=20):
    """Simple term frequency"""
    all_words = []
    
    for review in reviews:
        if pd.isna(review):
            continue
        
        # Basic preprocessing with regex
        words = re.findall(r'\b[a-zA-Z]{3,}\b', review.lower())
        
        # Words filtering
        filtered_words = {'the', 'and', 'was', 'were', 'are', 'had', 'have', 'has', 
                       'that', 'this', 'with', 'for', 'but', 'not', 'you', 'all',
                       'can', 'her', 'his', 'our', 'out', 'day', 'get', 'use',
                       'man', 'new', 'now', 'way', 'may', 'say', 'each', 'which',}
        
        words = [word for word in words if word not in filtered_words]
        all_words.extend(words)
    
    return Counter(all_words).most_common(top_n)

print("Positive Reviews - Top 50:")
simple_pos = simple_term_frequency(positive_reviews, 50)
for term, freq in simple_pos:
    print(f"  {term:<15} : {freq:4d}")

print("\nNegative Reviews - Top 50:")
simple_neg = simple_term_frequency(negative_reviews, 50)
for term, freq in simple_neg:
    print(f"  {term:<15} : {freq:4d}")