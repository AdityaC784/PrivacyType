import os
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
import pickle
import math


nltk.download('punkt', quiet=True)

class DataProcessor:
    def __init__(self, max_sequence_length=10, min_word_frequency=2):
        self.max_sequence_length = max_sequence_length
        self.min_word_frequency = min_word_frequency
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = {}
        self.vocabulary_size = 0
        
    def preprocess_text(self, text):
        """Preprocess text by converting to lowercase and removing special characters."""
        # Convert to string if not already a string
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def build_vocabulary(self, texts):
        """Build vocabulary from texts."""
        # Count word frequencies
        for text in texts:
            words = word_tokenize(self.preprocess_text(text))
            for word in words:
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
        
        # Filter words by frequency
        filtered_words = [word for word, count in self.word_counts.items() 
                         if count >= self.min_word_frequency]
        
        # Create word-to-index and index-to-word mappings
        self.word_to_index['<PAD>'] = 0  # Padding token
        self.word_to_index['<UNK>'] = 1  # Unknown token
        self.word_to_index['<START>'] = 2  # Start of sequence token
        self.word_to_index['<END>'] = 3  # End of sequence token
        
        # Add filtered words
        for i, word in enumerate(filtered_words):
            self.word_to_index[word] = i + 4  # +4 for special tokens
        
        # Create reverse mapping
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.vocabulary_size = len(self.word_to_index)
        
        print(f"Vocabulary size: {self.vocabulary_size}")
        return self.word_to_index, self.index_to_word
    
    def create_sequences(self, texts):
        """Create sequences of tokens from texts."""
        sequences = []
        next_words = []
        
        for text in texts:
            words = word_tokenize(self.preprocess_text(text))
            # Only add START token, not END token
            words = ['<START>'] + words
            
            for i in range(1, len(words)):
                # Get the sequence up to the current word
                seq_length = min(i, self.max_sequence_length)
                seq = words[max(0, i-seq_length):i]
                
                # Pad sequence if needed
                if len(seq) < self.max_sequence_length:
                    seq = ['<PAD>'] * (self.max_sequence_length - len(seq)) + seq
                
                # Convert words to indices
                seq_indices = [self.word_to_index.get(word, self.word_to_index['<UNK>']) for word in seq]
                
                # Skip if the next word is END token
                if i < len(words) - 1:  # Not the last word
                    next_word = words[i]
                    next_word_index = self.word_to_index.get(next_word, self.word_to_index['<UNK>'])
                    
                    # Only add sequences where next word is not END
                    if next_word != '<END>':
                        sequences.append(seq_indices)
                        next_words.append(next_word_index)
        
        return np.array(sequences), np.array(next_words)
    
    def prepare_data(self, texts, rebuild_vocab=False):
        """Prepare data by building vocabulary and creating sequences."""
        # Only build vocabulary if it's empty or explicitly requested
        if rebuild_vocab or not self.word_to_index:
            self.build_vocabulary(texts)
            
        sequences, next_words = self.create_sequences(texts)
        
        # Return sequences and integer labels (no one-hot encoding)
        return sequences, next_words
    
    def load_csv_data(self, file_path):
        """Load conversation data from a CSV file."""
        df = pd.read_csv(file_path)
        # Handle both 'text' and 'Text' column names
        column_name = 'Text' if 'Text' in df.columns else 'text'
        conversations = df[column_name].tolist()
        return conversations
        
    def load_multiple_csv_data(self, file_paths):
        """Load conversation data from multiple CSV files."""
        all_conversations = []
        for file_path in file_paths:
            conversations = self.load_csv_data(file_path)
            all_conversations.extend(conversations)
        return all_conversations
        
    def create_clients_from_data(self, sequences, next_words_one_hot, num_clients):
        """Dynamically create clients with equal data distribution."""
        total_samples = len(sequences)
        samples_per_client = math.ceil(total_samples / num_clients)
        
        client_data = []
        
        for c in range(num_clients):
            client_start = c * samples_per_client
            client_end = min((c + 1) * samples_per_client, total_samples)
            
            # If we've reached the end of the data, break
            if client_start >= total_samples:
                break
                
            client_sequences = sequences[client_start:client_end]
            client_next_words = next_words_one_hot[client_start:client_end]
            
            client_data.append({
                'client_id': f'client_{c+1}',
                'sequences': client_sequences,
                'next_words': client_next_words
            })
        
        return client_data
        
    def split_data_for_federated_learning(self, sequences, next_words, num_rounds=4, num_clients=2):
        """Split data into rounds and dynamically allocate to clients."""
    
        data_per_round = len(sequences) // num_rounds
        round_data = []
        
        for r in range(num_rounds):
            start_idx = r * data_per_round
            end_idx = (r + 1) * data_per_round if r < num_rounds - 1 else len(sequences)
            
            round_sequences = sequences[start_idx:end_idx]
            round_next_words = next_words[start_idx:end_idx]
            
            # Dynamically create clients with equal data distribution
            client_data = self.create_clients_from_data(
                round_sequences, round_next_words, num_clients
            )
            
            round_data.append({
                'round': r + 1,
                'client_data': client_data
            })
        
        return round_data
    
    def save(self, file_path='data_processor.pkl'):
        """Save the data processor state."""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'max_sequence_length': self.max_sequence_length,
                'min_word_frequency': self.min_word_frequency,
                'word_to_index': self.word_to_index,
                'index_to_word': self.index_to_word,
                'word_counts': self.word_counts,
                'vocabulary_size': self.vocabulary_size,
            }, f)
    
    @classmethod
    def load(cls, file_path='data_processor.pkl'):
        """Load a data processor from a file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        processor = cls(
            max_sequence_length=data['max_sequence_length'],
            min_word_frequency=data['min_word_frequency']
        )
        processor.word_to_index = data['word_to_index']
        processor.index_to_word = data['index_to_word']
        processor.word_counts = data['word_counts']
        processor.vocabulary_size = data['vocabulary_size']
        
        return processor
    
    def tokenize_input(self, text):
        """Tokenize input text for prediction."""
        words = word_tokenize(self.preprocess_text(text))
        
        # Get the last max_sequence_length words
        seq_length = min(len(words), self.max_sequence_length)
        seq = words[-seq_length:]
        
        # Pad sequence if needed
        if len(seq) < self.max_sequence_length:
            seq = ['<PAD>'] * (self.max_sequence_length - len(seq)) + seq
        
        # Convert words to indices
        seq_indices = [self.word_to_index.get(word, self.word_to_index['<UNK>']) for word in seq]
        
        return np.array([seq_indices])

    def predict_next_word(self, model, text, top_n=5):
        """Predict the next word, excluding special tokens from predictions."""
        input_sequence = self.tokenize_input(text)
        indices, probabilities = model.predict(input_sequence, top_n=top_n + 3)  # Get more predictions to filter
        
        # Filter out special tokens and get top_n predictions
        filtered_predictions = []
        for idx, prob in zip(indices, probabilities):
            word = self.index_to_word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<UNK>', '<START>', '<END>']:
                filtered_predictions.append((word, prob))
                if len(filtered_predictions) >= top_n:
                    break
        
        return filtered_predictions