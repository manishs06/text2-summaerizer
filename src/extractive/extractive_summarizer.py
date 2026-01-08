import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import networkx as nx
from ..preprocessing.text_preprocessor import TextPreprocessor

class ExtractiveSummarizer:
    """
    A class for performing extractive text summarization using various algorithms.
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def tfidf_summarize(self, text, num_sentences=3):
        """
        Summarize text using TF-IDF approach.
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in the summary
            
        Returns:
            str: Summarized text
        """
        # Preprocess the text
        processed = self.preprocessor.preprocess(text)
        sentences = processed['sentences']
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores based on TF-IDF
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get indices of top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices.sort()  # Sort to maintain original order
        
        # Create summary
        summary_sentences = [sentences[i] for i in top_indices]
        return ' '.join(summary_sentences)
    
    def textrank_summarize(self, text, num_sentences=3, damping=0.85, max_iter=50):
        """
        Summarize text using TextRank algorithm (similar to PageRank).
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in the summary
            damping (float): Damping factor for PageRank algorithm
            max_iter (int): Maximum number of iterations
            
        Returns:
            str: Summarized text
        """
        # Preprocess the text
        processed = self.preprocessor.preprocess(text)
        sentences = processed['sentences']
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Create similarity matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity between sentences
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create graph from similarity matrix
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank algorithm
        scores = nx.pagerank(graph, max_iter=max_iter, tol=1e-4)
        
        # Get top sentences based on scores
        ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in ranked_sentences[:num_sentences]]
        top_indices.sort()  # Sort to maintain original order
        
        # Create summary
        summary_sentences = [sentences[i] for i in top_indices]
        return ' '.join(summary_sentences)
    
    def frequency_based_summarize(self, text, num_sentences=3):
        """
        Summarize text using word frequency approach.
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in the summary
            
        Returns:
            str: Summarized text
        """
        # Preprocess the text
        processed = self.preprocessor.preprocess(text)
        sentences = processed['sentences']
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Calculate word frequencies
        all_words = processed['all_words']
        word_freq = self.preprocessor.get_word_frequency(all_words)
        
        # Score sentences based on word frequencies
        sentence_scores = []
        for sentence in sentences:
            words = self.preprocessor.tokenize_words(sentence)
            words = [word.lower() for word in words if word not in '.!?,' and word.lower() not in self.preprocessor.stop_words]
            
            if len(words) == 0:
                sentence_scores.append(0)
                continue
                
            score = sum(word_freq.get(word, 0) for word in words) / len(words)
            sentence_scores.append(score)
        
        # Get top sentences
        top_indices = np.array(sentence_scores).argsort()[-num_sentences:][::-1]
        top_indices.sort()  # Sort to maintain original order
        
        # Create summary
        summary_sentences = [sentences[i] for i in top_indices]
        return ' '.join(summary_sentences)
    
    def summarize(self, text, method='textrank', num_sentences=3):
        """
        Main method to summarize text using specified method.
        
        Args:
            text (str): Input text to summarize
            method (str): Method to use ('tfidf', 'textrank', 'frequency')
            num_sentences (int): Number of sentences in the summary
            
        Returns:
            str: Summarized text
        """
        if method == 'tfidf':
            return self.tfidf_summarize(text, num_sentences)
        elif method == 'textrank':
            return self.textrank_summarize(text, num_sentences)
        elif method == 'frequency':
            return self.frequency_based_summarize(text, num_sentences)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tfidf', 'textrank', or 'frequency'")