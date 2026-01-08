try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

from ..preprocessing.text_preprocessor import TextPreprocessor

class AbstractiveSummarizer:
    """
    A class for performing abstractive text summarization using transformer models.
    """
    
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the abstractive summarizer with a pre-trained transformer model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        
        if pipeline is None:
            print("Warning: transformers.pipeline not available. Abstractive summarization will use fallback.")
            self.summarizer = None
        else:
            try:
                # Initialize using the pipeline approach
                self.summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    tokenizer=model_name
                )
            except Exception as e:
                print(f"Could not load {model_name}, using default model: {e}")
                try:
                    # Fallback to a simpler model
                    self.summarizer = pipeline(
                        "summarization",
                        model="sshleifer/distilbart-cnn-12-6"
                    )
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    raise
    
    def summarize(self, text, max_length=130, min_length=30, do_sample=False):
        """
        Summarize text using the transformer model.
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            do_sample (bool): Whether to use sampling instead of greedy decoding
            
        Returns:
            str: Summarized text
        """
        # Clean the input text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # If the text is too short, return as is
        if len(cleaned_text.split()) < min_length:
            return cleaned_text
        
        if self.summarizer is None:
            # Fallback: if the model is not available, return a simple extractive summary
            print("Using fallback extractive summarization...")
            
            # Use extractive summarization as fallback
            from ..extractive.extractive_summarizer import ExtractiveSummarizer
            fallback_summarizer = ExtractiveSummarizer()
            return fallback_summarizer.summarize(cleaned_text, method='textrank', num_sentences=3)
        
        try:
            # Generate summary
            summary_result = self.summarizer(
                cleaned_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )
            
            # Extract the summary text
            summary = summary_result[0]['summary_text']
            return summary
            
        except Exception as e:
            # Fallback: if the model fails, return a simple extractive summary
            print(f"Transformer model failed: {e}")
            print("Falling back to extractive summarization...")
            
            # Use extractive summarization as fallback
            from ..extractive.extractive_summarizer import ExtractiveSummarizer
            fallback_summarizer = ExtractiveSummarizer()
            return fallback_summarizer.summarize(cleaned_text, method='textrank', num_sentences=3)
    
    def batch_summarize(self, texts, max_length=130, min_length=30, do_sample=False):
        """
        Summarize multiple texts at once.
        
        Args:
            texts (list): List of input texts to summarize
            max_length (int): Maximum length of each summary
            min_length (int): Minimum length of each summary
            do_sample (bool): Whether to use sampling instead of greedy decoding
            
        Returns:
            list: List of summarized texts
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text, max_length, min_length, do_sample)
            summaries.append(summary)
        return summaries