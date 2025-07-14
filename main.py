import os
import numpy as np
import argparse
from data_processor import DataProcessor
from lstm_model import LSTMModel
from federated_learning import FederatedLearning
import matplotlib.pyplot as plt
import nltk


def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Next Word Prediction')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of clients for federated learning')
    parser.add_argument('--num_rounds', type=int, default=4, help='Number of federated learning rounds')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs per round')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--validation_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--max_sequence_length', type=int, default=8, help='Maximum sequence length')
    parser.add_argument('--min_word_frequency', type=int, default=2, help='Minimum word frequency')
    return parser.parse_args()


nltk.download('punkt')
nltk.download('punkt_tab')
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)


# Parse command line arguments
args = parse_arguments()

# Initialize data processor
data_processor = DataProcessor(max_sequence_length=args.max_sequence_length, 
                              min_word_frequency=args.min_word_frequency)

print("Loading training data...")
train_conversations = data_processor.load_csv_data('train_data.csv')
print(f"Total number of training conversations: {len(train_conversations)}")

print("Building vocabulary...")
data_processor.build_vocabulary(train_conversations)

print("Creating sequences and splitting into rounds...")
sequences, next_words = data_processor.prepare_data(train_conversations, rebuild_vocab=False)
round_data = data_processor.split_data_for_federated_learning(
    sequences, next_words, num_rounds=args.num_rounds, num_clients=args.num_clients
)

print(f"Data split into {len(round_data)} rounds with {len(round_data[0]['client_data'])} clients each")


data_processor.save('models/data_processor.pkl')


federated = FederatedLearning(
    vocabulary_size=data_processor.vocabulary_size,
    embedding_dim=128,
    lstm_units=256,
    dropout_rate=0.2
)


epochs_per_round = args.epochs
batch_size = args.batch_size
validation_split = args.validation_split

# Train for multiple rounds
all_training_logs = []

for round_info in round_data:
    print(f"\n==== Starting Round {round_info['round']} ====")
    
 
    round_result = federated.train_round(
        round_info,
        epochs=epochs_per_round,
        batch_size=batch_size,
        validation_split=validation_split
    )
    
    all_training_logs.extend(round_result['training_logs'])
    
    # Save the current state
    federated.save(directory='models/federated')
    
    # Plot and save the current training metrics
    fig = federated.plot_training_metrics(
        show=False,
        save_path=f"visualizations/round_{round_info['round']}_metrics.png"
    )
    plt.close(fig)
    
    print(f"==== Completed Round {round_info['round']} ====\n")


federated.save(directory='models/federated_final')


federated.plot_training_metrics(
    show=True,
    save_path="visualizations/final_metrics.png"
)

# Load test data for evaluation
print("\n==== Evaluating on Test Data ====\n")

test_conversations = data_processor.load_csv_data('test_data.csv')
print(f"Total number of test conversations: {len(test_conversations)}")

# Create sequences from test data
test_sequences, test_next_words = data_processor.create_sequences(test_conversations)


test_loss, test_accuracy = federated.global_model.evaluate(test_sequences, test_next_words)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


print("\n==== Example Predictions ====\n")

test_inputs = [
    "hello how are you",
    "i am going to",
    "what school do you",
    "thank you for"
]

for test_input in test_inputs:
   
    predictions = data_processor.predict_next_word(
        federated.global_model,
        test_input,
        top_n=5
    )
    
    print(f"\nInput: '{test_input}'")
    print("Predicted next words:")
    for word, prob in predictions:
        print(f"  {word}: {prob:.4f}")

print("\nFederated Learning Next Word Prediction System completed successfully!")
print(f"Trained with {args.num_clients} clients over {args.num_rounds} rounds")