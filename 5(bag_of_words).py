import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download NLTK resources if not already installed
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Fetch English stopwords
    stop_words = set(stopwords.words('english'))
    # Tokenize text
    word_tokens = word_tokenize(text)
    # Remove non-alphabetical tokens and stopwords
    filtered_words = [word.lower() for word in word_tokens if word.isalpha() and word.lower() not in stop_words]
    return filtered_words

def create_bow_model(texts):
    all_words = []
    # Process each text
    for text in texts:
        words = preprocess_text(text)
        all_words.extend(words)
    # Calculate frequency distribution
    word_freq = FreqDist(all_words)
    # Create Bag of Words model
    bow_model = {word: freq for word, freq in word_freq.items()}
    return bow_model

# Example usage
texts = [
    "The cat sat on the mat, and the mat was comfortable.",
    "She sang a sweet song, a song that touched everyone's heart.",
    "Coding can be challenging, but coding is also incredibly rewarding."
]

# Generate Bag of Words model
bow_model = create_bow_model(texts)

# Print the Bag of Words model
print("Bag of Words Model:")
for word, freq in bow_model.items():
    print(f"{word}: {freq}")
