# PrivacyType: Federated Learning for Smart Text Prediction

A privacy-preserving next word prediction system using federated learning, inspired by Google's Gboard implementation.

## Overview

This project implements a federated learning system for next word prediction, where multiple clients can train a shared model without sharing their raw data. The system uses LSTM-based neural networks to predict the next word in a sequence while maintaining user privacy.

## Features

- **Privacy-Preserving Training**: Train models without sharing raw text data
- **Federated Learning**: Collaborative model training across multiple clients
- **LSTM Architecture**: Advanced neural network for sequence prediction
- **Web Interface**: Flask-based UI for training and prediction
- **Real-time Predictions**: Get top 5 most likely next words
- **Training Visualization**: Track and visualize training progress

## Architecture

- **Data Processing**: Text tokenization, vocabulary building, and sequence creation
- **Model Architecture**: LSTM with embedding layer for word representation
- **Federated Learning**: Weight aggregation and model distribution
- **Web Application**: User interface for training and prediction

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/AdityaC784/PrivacyType.git
   cd PrivacyType
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

1. **Prepare Data**:
   - Place training data in CSV format in the `client_data` directory
   - Each CSV should have a 'Text' or 'text' column

2. **Train Model**:
   - Run the web application: `python app.py`
   - Navigate to the training page
   - Configure training parameters (clients, rounds, epochs)
   - Start training

3. **Make Predictions**:
   - Use the prediction interface to enter text
   - Get the top 5 most likely next words

## Project Structure

```
privacytype/
├── app.py                 # Flask web application
├── data_processor.py      # Text processing and sequence creation
├── lstm_model.py          # LSTM model implementation
├── federated_learning.py  # Federated learning logic
├── main.py                # Main training script
├── run_federated.py       # Federated learning runner
├── client_data/           # Directory for client datasets
├── models/                # Saved models and metadata
└── visualizations/        # Training visualization outputs
```

## Files to Exclude from Version Control

The following files and directories should not be uploaded to version control:

1. **Model files and directories**:
   - `models/` directory (contains trained models and weights)
   - `*.h5` files (model weight files)
   - `*.pkl` files (pickle files with model state)

2. **Generated data**:
   - `visualizations/` directory (contains generated plots)
   - `training_logs.json` (runtime logs)

3. **Environment and system files**:
   - `__pycache__/` directories
   - `*.pyc` files
   - `.DS_Store` (Mac system files)
   - `venv/` or `env/` (virtual environment directories)

4. **Large data files**:
   - `client_data/` directory (contains your training datasets)
   - Any large CSV files

Instead, include sample data files or instructions for obtaining them in the repository.

## Technical Details

- **Vocabulary Size**: 11,418 words (configurable)
- **Embedding Dimension**: 128
- **LSTM Units**: 256
- **Sequence Length**: 10 tokens
- **Dropout Rate**: 0.2

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Google's Gboard implementation of federated learning
- Uses TensorFlow and Keras for deep learning
- Flask for web interface 