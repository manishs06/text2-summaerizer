import math
from collections import Counter, defaultdict
import networkx as nx
from ..preprocessing.text_preprocessor import TextPreprocessor

class ExtractiveSummarizer:
    """
    A class for performing extractive text summarization using various algorithms.
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def _get_tfidf_scores(self, sentences):
        """
        Calculate TF-IDF scores for a list of sentences without using scikit-learn.
        """
        # Tokenize sentences into words
        tokenized_sentences = []
        for sent in sentences:
            words = self.preprocessor.tokenize_words(sent)
            words = [w.lower() for w in words if w.isalnum() and w.lower() not in self.preprocessor.stop_words]
            tokenized_sentences.append(words)

        # Calculate TF
        tfs = []
        for words in tokenized_sentences:
            tfs.append(Counter(words))

        # Calculate IDF
        num_docs = len(sentences)
        idf = defaultdict(float)
        all_words = set([word for words in tokenized_sentences for word in words])
        
        for word in all_words:
            num_docs_with_word = sum(1 for words in tokenized_sentences if word in words)
            idf[word] = math.log(num_docs / (1 + num_docs_with_word))

        # Calculate TF-IDF matrix (as a list of dictionaries for sparsity)
        tfidf_matrix = []
        for tf in tfs:
            tfidf = {}
            for word, freq in tf.items():
                tfidf[word] = freq * idf[word]
            tfidf_matrix.append(tfidf)
            
        return tfidf_matrix, all_words

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two sparse vectors (dictionaries)."""
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
    
    def tfidf_summarize(self, text, num_sentences=3):
        """
        Summarize text using TF-IDF approach without numpy or scikit-learn.
        """
        # Preprocess the text
        processed = self.preprocessor.preprocess(text)
        sentences = processed['sentences']
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Get TF-IDF matrix
        tfidf_matrix, _ = self._get_tfidf_scores(sentences)
        
        # Calculate sentence scores (sum of TF-IDF values in each sentence)
        sentence_scores = []
        for vec in tfidf_matrix:
            sentence_scores.append(sum(vec.values()))
        
        # Get indices of top sentences
        top_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
        top_indices.sort()  # Sort to maintain original order
        
        # Create summary
        summary_sentences = [sentences[i] for i in top_indices]
        return ' '.join(summary_sentences)
    
    def textrank_summarize(self, text, num_sentences=3, damping=0.85, max_iter=50):
        """
        Summarize text using TextRank algorithm (similar to PageRank) without scikit-learn.
        """
        # Preprocess the text
        processed = self.preprocessor.preprocess(text)
        sentences = processed['sentences']
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Get TF-IDF matrix
        tfidf_matrix, _ = self._get_tfidf_scores(sentences)
        
        # Create graph
        graph = nx.Graph()
        num_sentences_total = len(sentences)
        
        for i in range(num_sentences_total):
            for j in range(i + 1, num_sentences_total):
                similarity = self._cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
                if similarity > 0:
                    graph.add_edge(i, j, weight=similarity)
        
        # Apply PageRank algorithm
        try:
            scores = nx.pagerank(graph, alpha=damping, max_iter=max_iter, tol=1e-4)
        except:
            # Fallback if pagerank fails to converge
            scores = {i: 0 for i in range(num_sentences_total)}
        
        # Get top sentences based on scores
        ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in ranked_sentences[:num_sentences]]
        top_indices.sort()  # Sort to maintain original order
        
        # Create summary
        summary_sentences = [sentences[i] for i in top_indices]
        return ' '.join(summary_sentences)
    
    def frequency_based_summarize(self, text, num_sentences=3):
        """
        Summarize text using word frequency approach without numpy.
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
            words = [word.lower() for word in words if word.isalnum() and word.lower() not in self.preprocessor.stop_words]
            
            if len(words) == 0:
                sentence_scores.append(0)
                continue
                
            score = sum(word_freq.get(word, 0) for word in words) / len(words)
            sentence_scores.append(score)
        
        # Get top sentences
        top_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
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