import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

# Configure NLTK to use /tmp for data if on Vercel (read-only filesystem)
import os
nltk_data_path = os.path.join('/tmp', 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
def download_nltk_data():
    for package in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            nltk.download(package, download_dir=nltk_data_path)

download_nltk_data()

class TextPreprocessor:
    """
    A class for preprocessing text data for text summarization tasks.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """
        Clean the input text by removing special characters, extra whitespaces, etc.
        
        Args:
            text (str): Input text to be cleaned
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        # Strip leading/trailing whitespaces
        text = text.strip()
        
        return text
    
    def tokenize_sentences(self, text):
        """
        Tokenize text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        sentences = sent_tokenize(text)
        return sentences
    
    def tokenize_words(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of words
        """
        words = word_tokenize(text)
        return words
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens without stopwords
        """
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return filtered_tokens
    
    def stem_words(self, tokens):
        """
        Apply stemming to tokens.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return stemmed_tokens
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing preprocessed components
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize into sentences
        sentences = self.tokenize_sentences(cleaned_text)
        
        # Tokenize into words and process
        processed_sentences = []
        all_words = []
        
        for sentence in sentences:
            words = self.tokenize_words(sentence)
            # Convert to lowercase
            words = [word.lower() for word in words if word not in string.punctuation]
            # Remove stopwords
            words = self.remove_stopwords(words)
            # Apply stemming
            words = self.stem_words(words)
            
            processed_sentences.append(words)
            all_words.extend(words)
        
        return {
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'processed_sentences': processed_sentences,
            'all_words': all_words
        }
    
    def get_word_frequency(self, tokens):
        """
        Calculate frequency of each word in the token list.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            dict: Dictionary with word frequencies
        """
        word_freq = {}
        for word in tokens:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        return word_freq