import numpy as np
import os
import json
import tensorflow as tf
from lstm_model import LSTMModel
import matplotlib.pyplot as plt
from datetime import datetime

class FederatedLearning:
    def __init__(self, vocabulary_size, embedding_dim=64, lstm_units=128, dropout_rate=0.2):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.global_model = LSTMModel(
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            name='global_model'
        )
        self.client_models = {}
        self.round_history = []
        self.current_round = 0
    
    def create_client_model(self, client_id):
        """Create a model for a client."""
        model = LSTMModel(
            vocabulary_size=self.vocabulary_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            name=f'client_{client_id}'
        )
        self.client_models[client_id] = model
        return model
    
    def train_client(self, client_id, X_train, y_train, epochs=5, batch_size=32, validation_split=0.1):
        """Train a client model."""
        if client_id not in self.client_models:
            self.create_client_model(client_id)
        
        # Set the client model weights to the global model weights
        if self.current_round > 0:
            self.client_models[client_id].set_weights(self.global_model.get_weights())
        
        # Store client data sizes for weighted aggregation
        if not hasattr(self, 'client_data_sizes'):
            self.client_data_sizes = {}
        self.client_data_sizes[client_id] = len(X_train)
        
        # Define a callback to store training metrics
        class TrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self, client_id, round_num):
                super().__init__()
                self.client_id = client_id
                self.round_num = round_num
                self.logs = []
                
            def on_epoch_end(self, epoch, logs=None):
                self.logs.append({
                    'round': self.round_num,
                    'client_id': self.client_id,
                    'epoch': epoch + 1,
                    'loss': float(logs['loss']),
                    'accuracy': float(logs['accuracy']),
                    'val_loss': float(logs['val_loss']),
                    'val_accuracy': float(logs['val_accuracy'])
                })
        
        callback = TrainingCallback(client_id, self.current_round + 1)
        
        # Train the client model
        history = self.client_models[client_id].train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[callback]
        )
        
        return history, callback.logs
    
    def aggregate_models(self, weighted=True):
        """Aggregate client models to update the global model."""
        if not self.client_models:
            raise ValueError("No client models available for aggregation.")
        
        # Get the weights of all client models
        client_weights = [model.get_weights() for model in self.client_models.values()]
        
        if weighted and hasattr(self, 'client_data_sizes') and self.client_data_sizes:
            # Calculate weights based on data size
            total_samples = sum(self.client_data_sizes.values())
            weight_factors = {client_id: size/total_samples 
                             for client_id, size in self.client_data_sizes.items()}
            
            # Apply weighted averaging
            client_ids = list(self.client_models.keys())
            weighted_avg = []
            
            for i in range(len(client_weights[0])):
                layer_weights = np.zeros_like(client_weights[0][i])
                for j, client_id in enumerate(client_ids):
                    layer_weights += client_weights[j][i] * weight_factors[client_id]
                weighted_avg.append(layer_weights)
                
            avg_weights = weighted_avg
        else:
            # Calculate the simple average weights
            avg_weights = [np.mean([client_weight[i] for client_weight in client_weights], axis=0)
                          for i in range(len(client_weights[0]))]
        
        # Update the global model with the average weights
        self.global_model.set_weights(avg_weights)
        
        # Update round information
        self.current_round += 1
        
        # Store round information
        round_info = {
            'round': self.current_round,
            'timestamp': datetime.now().isoformat(),
            'num_clients': len(self.client_models),
            'client_ids': list(self.client_models.keys()),
            'aggregation_method': 'weighted' if weighted else 'simple'
        }
        self.round_history.append(round_info)
        
        return round_info
    
    def train_round(self, round_data, epochs=5, batch_size=32, validation_split=0.1):
        """Train a complete federated round."""
        round_num = round_data['round']
        client_training_logs = []
        
        print(f"\n--- Starting Federated Learning Round {round_num} ---")
        
        # Clear previous client models if the number of clients has changed
        num_clients = len(round_data['client_data'])
        current_clients = set(self.client_models.keys())
        expected_clients = set([client_data['client_id'] for client_data in round_data['client_data']])
        
        # If client set has changed, reset client models
        if current_clients != expected_clients:
            print(f"Client set has changed. Resetting client models.")
            self.client_models = {}
        
        # Train each client model
        for client_data in round_data['client_data']:
            client_id = client_data['client_id']
            sequences = client_data['sequences']
            next_words = client_data['next_words']
            
            print(f"\nTraining client {client_id} with {len(sequences)} examples")
            
            # Train the client model
            _, logs = self.train_client(
                client_id,
                sequences, next_words,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split
            )
            
            client_training_logs.extend(logs)
        
        # Aggregate client models
        aggregation_info = self.aggregate_models()
        
        print(f"\n--- Completed Federated Learning Round {round_num} ---")
        print(f"Global model updated with weights from {len(self.client_models)} clients")
        
        # Return combined logs
        return {
            'round_info': aggregation_info,
            'training_logs': client_training_logs
        }
    

    
    def save(self, directory='federated_models'):
        """Save the federated learning state."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the global model
        self.global_model.save(os.path.join(directory, 'global'))
        
        # Save the client models
        for client_id, model in self.client_models.items():
            model.save(os.path.join(directory, client_id))
        
        # Save the federated learning metadata
        metadata = {
            'vocabulary_size': self.vocabulary_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'current_round': self.current_round,
            'round_history': self.round_history,
            'client_ids': list(self.client_models.keys())
        }
        
        with open(os.path.join(directory, 'federated_metadata.json'), 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, directory='federated_models'):
        """Load a federated learning state from saved files."""
        # Load the metadata
        with open(os.path.join(directory, 'federated_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Create a new federated learning instance
        federated = cls(
            vocabulary_size=metadata['vocabulary_size'],
            embedding_dim=metadata['embedding_dim'],
            lstm_units=metadata['lstm_units'],
            dropout_rate=metadata['dropout_rate']
        )
        
        # Load the global model
        federated.global_model = LSTMModel.load(
            directory=os.path.join(directory, 'global'),
            name='global_model'
        )
        
        # Load the client models
        for client_id in metadata['client_ids']:
            federated.client_models[client_id] = LSTMModel.load(
                directory=os.path.join(directory, client_id),
                name=f'client_{client_id}'
            )
        
        # Load the federated learning state
        federated.current_round = metadata['current_round']
        federated.round_history = metadata['round_history']
        
        return federated
    
    def plot_training_metrics(self, show=True, save_path=None):
        """Plot training metrics across rounds."""
        if not self.round_history:
            print("No training history available.")
            return
        
        # Collect metrics from all client models
        metrics_by_round = {}
        for client_id, model in self.client_models.items():
            for entry in model.training_history:
                round_key = entry.get('round', 1)  # Default to round 1 if not specified
                if round_key not in metrics_by_round:
                    metrics_by_round[round_key] = {'loss': [], 'accuracy': []}
                
                metrics_by_round[round_key]['loss'].append(entry['loss'])
                metrics_by_round[round_key]['accuracy'].append(entry['accuracy'])
        
        # Create the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss by round
        rounds = sorted(metrics_by_round.keys())
        avg_loss = [np.mean(metrics_by_round[r]['loss']) for r in rounds]
        ax1.plot(rounds, avg_loss, 'o-', label='Average Loss')
        ax1.set_title('Loss by Round')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy by round
        avg_accuracy = [np.mean(metrics_by_round[r]['accuracy']) for r in rounds]
        ax2.plot(rounds, avg_accuracy, 'o-', label='Average Accuracy')
        ax2.set_title('Accuracy by Round')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        if show:
            plt.show()
        
        return fig