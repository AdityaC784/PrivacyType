import tensorflow as tf
import numpy as np
import os
import json
import pickle

class LSTMModel:
    def __init__(self, vocabulary_size, embedding_dim=64, lstm_units=128, dropout_rate=0.2, name='client_model'):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.name = name
        self.model = self._build_model()
     
        self.model.build(input_shape=(None, 10))  # 10 is the max_sequence_length
        self.training_history = []
    
    def _build_model(self):
        """Build the LSTM model for next word prediction."""
        model = tf.keras.Sequential(name=self.name)
        
     
        model.add(tf.keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_dim,
            mask_zero=True, 
            name='embedding'
        ))
        
       
        model.add(tf.keras.layers.LSTM(
            units=self.lstm_units,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            return_sequences=False,
            name='lstm'
        ))
        
      
        model.add(tf.keras.layers.Dropout(self.dropout_rate, name='dropout'))
        
       
        model.add(tf.keras.layers.Dense(
            units=self.vocabulary_size,
            activation='softmax',
            name='output'
        ))
      
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,  # Explicit learning rate
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=None):
        """Train the model."""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history
        for i in range(len(history.history['loss'])):
            self.training_history.append({
                'epoch': i + 1,
                'loss': float(history.history['loss'][i]),
                'accuracy': float(history.history['accuracy'][i]),
                'val_loss': float(history.history['val_loss'][i]),
                'val_accuracy': float(history.history['val_accuracy'][i])
            })
        
        return history
    
    def predict(self, X, top_n=5):
        """Predict the next word probabilities."""
        predictions = self.model.predict(X)
        
        # Get the indices of the top n predictions
        top_indices = np.argsort(predictions[0])[-top_n:][::-1]
        
        # Get the probabilities of the top n predictions
        top_probs = predictions[0][top_indices]
        
        return top_indices, top_probs
    
    def get_weights(self):
       
        return self.model.get_weights()
    
    def set_weights(self, weights):
      
        self.model.set_weights(weights)
    
    def save(self, directory='models'):
        """Save the model and its metadata."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the model weights with correct extension
        weights_path = os.path.join(directory, f"{self.name}.weights.h5")
        self.model.save_weights(weights_path)
        
        # Save the model architecture and training history
        metadata = {
            'name': self.name,
            'vocabulary_size': self.vocabulary_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'training_history': self.training_history
        }
        
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, directory='models', name='client_model'):
        """Load a model from its saved files."""
       
        with open(os.path.join(directory, f"{name}_metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Create a new model with the same architecture
        model = cls(
            vocabulary_size=metadata['vocabulary_size'],
            embedding_dim=metadata['embedding_dim'],
            lstm_units=metadata['lstm_units'],
            dropout_rate=metadata['dropout_rate'],
            name=metadata['name']
        )
        
        # Load the weights
        model.model.load_weights(os.path.join(directory, f"{name}.weights.h5"))
        
        # Load the training history
        model.training_history = metadata['training_history']
        
        return model
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        return self.model.evaluate(X_test, y_test)