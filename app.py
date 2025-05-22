from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response, flash
import pandas as pd
import os
import json
import threading
from data_processor import DataProcessor
from federated_learning import FederatedLearning
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('client_data', exist_ok=True)

def get_client_files():
    """Get list of available client data files."""
    return [f for f in os.listdir('client_data') if f.endswith('.csv')]

@app.route('/')
def index():
    """Home page showing system status and options."""
    client_files = get_client_files()
    model_exists = os.path.exists('models/federated_final')
    
    return render_template('index.html', 
                         client_files=client_files,
                         model_exists=model_exists)

@app.route('/view_data/<filename>')
def view_data(filename):
    """View contents of a client data file."""
    try:
        # Read the CSV file
        df = pd.read_csv(os.path.join('client_data', filename))
        
        # Check if 'text' or 'Text' column exists, otherwise ensure a text column exists
        if 'text' in df.columns:
            text_column = 'text'
        elif 'Text' in df.columns:
            text_column = 'Text'
        else:
            # If no text column is found, use the first column as text
            text_column = df.columns[0]
            df = df.rename(columns={text_column: 'text'})
            text_column = 'text'
        
        # Convert data to records and add the text field explicitly
        records = df.head(100).to_dict('records')
        
        # Ensure each record has a 'text' field
        for record in records:
            if text_column != 'text' and text_column in record:
                record['text'] = record[text_column]
            elif 'text' not in record:
                record['text'] = "No text data available"
        
        return render_template('view_data.html', 
                             filename=filename,
                             data=records)
    except Exception as e:
        flash(f"Error viewing data: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/add_client', methods=['GET', 'POST'])
def add_client():
    """Add new client data."""
    if request.method == 'POST':
        try:
            # Check if file is included in the request
            if 'file' not in request.files:
                flash('No file part in the request', 'danger')
                return render_template('add_client.html')
                
            # Get the uploaded file
            file = request.files['file']
            
            # Check if a file was selected
            if file.filename == '':
                flash('No file selected', 'danger')
                return render_template('add_client.html')
                
            if file and file.filename.endswith('.csv'):
                # Save the file
                filename = f"client{len(get_client_files()) + 1}_data.csv"
                file_path = os.path.join('client_data', filename)
                file.save(file_path)
                
                # Verify the file was correctly saved
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    flash(f'File {filename} uploaded successfully!', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('File upload failed - could not verify saved file', 'danger')
                    return render_template('add_client.html')
            
            flash("Invalid file format. Please upload a CSV file.", 'warning')
            return render_template('add_client.html')
            
        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')
            return render_template('add_client.html')
    
    return render_template('add_client.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Train the federated learning model."""
    client_files = get_client_files()
    total_available_clients = len(client_files)
    
    if request.method == 'POST':
        try:
            import sys
            import subprocess
            # Get training parameters from form
            num_clients = int(request.form.get('num_clients', 2))
            epochs = int(request.form.get('epochs', 10))
            batch_size = int(request.form.get('batch_size', 32))
            num_rounds = int(request.form.get('num_rounds', 4))
            
            # Create training config file for progress tracking
            training_config = {
                "num_clients": num_clients,
                "epochs": epochs,
                "batch_size": batch_size,
                "num_rounds": num_rounds,
                "clients": client_files[:num_clients] if num_clients <= total_available_clients else client_files
            }
            
            # Ensure the models directory exists
            os.makedirs('models', exist_ok=True)
            
            # Save the training configuration
            with open('models/training_config.json', 'w') as f:
                json.dump(training_config, f)
            
            # Reset training logs
            with open('training_logs.json', 'w') as f:
                json.dump({
                    'current_round': 0,
                    'current_epoch': 0,
                    'validation_accuracy': 0.0,
                    'training_complete': False,
                    'status_message': 'Starting training...',
                    'history': []
                }, f)

            # Start training in a separate thread
            def train_async():
                try:
                    cmd = [
                        sys.executable, 'run_federated.py',
                        '--num_clients', str(num_clients),
                        '--num_rounds', str(num_rounds),
                        '--epochs', str(epochs),
                        '--batch_size', str(batch_size)
                    ]
                    subprocess.run(cmd, check=True)
                    
                    # Update status when training completes
                    with open('training_logs.json', 'r') as f:
                        logs = json.load(f)
                    
                    logs['training_complete'] = True
                    logs['status_message'] = 'Training completed successfully!'
                    
                    with open('training_logs.json', 'w') as f:
                        json.dump(logs, f)
                        
                except Exception as e:
                    print(f"Error during training: {e}")
                    # Update status on error
                    try:
                        with open('training_logs.json', 'r') as f:
                            logs = json.load(f)
                        
                        logs['status_message'] = f'Error during training: {str(e)}'
                        
                        with open('training_logs.json', 'w') as f:
                            json.dump(logs, f)
                    except:
                        pass

            thread = threading.Thread(target=train_async)
            thread.daemon = True
            thread.start()

            flash('Training started! Redirecting to progress page...', 'success')
            return redirect(url_for('training_progress'))
        except Exception as e:
            flash(f'Error starting training: {str(e)}', 'danger')
            return render_template('train.html', client_files=client_files)

    return render_template('train.html', client_files=client_files)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Make predictions using the trained model."""
    if not os.path.exists('models/federated_final'):
        flash('Model not trained yet. Please train the model first.', 'warning')
        return render_template('predict.html')
    
    if request.method == 'POST':
        try:
            text = request.form.get('text', '')
            if not text:
                flash('Please enter some text.', 'warning')
                return render_template('predict.html')
            
            # Load model and processor
            try:
                data_processor = DataProcessor.load('models/data_processor.pkl')
                federated = FederatedLearning.load('models/federated_final')
            except Exception as e:
                flash(f'Error loading model: {str(e)}', 'danger')
                return render_template('predict.html', text=text)
            
            # Get predictions
            try:
                predictions = data_processor.predict_next_word(
                    federated.global_model,
                    text,
                    top_n=5
                )
                
                return render_template('predict.html', 
                                     text=text,
                                     predictions=predictions)
            except Exception as e:
                flash(f'Error making prediction: {str(e)}', 'danger')
                return render_template('predict.html', text=text)
            
        except Exception as e:
            flash(str(e), 'danger')
            return render_template('predict.html')
    
    return render_template('predict.html')

@app.route('/training_progress')
def training_progress():
    """Show training progress and results."""
    try:
        # Load configuration
        config = {}
        try:
            with open('models/training_config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {
                "num_rounds": 0,
                "epochs": 0,
                "batch_size": 0,
                "clients": []
            }
        
        # Load logs
        try:
            with open('training_logs.json', 'r') as f:
                logs = json.load(f)
                
        except FileNotFoundError:
            logs = {
                'current_round': 0,
                'current_epoch': 0,
                'validation_accuracy': 0.0,
                'training_complete': False,
                'status_message': 'Waiting to start training...',
                'history': []
            }
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'logs': logs, 'config': config})
        
        # Add cache busting parameter to prevent browser caching
        response = make_response(render_template('training_progress.html',
                                              config=config,
                                              logs=logs))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        app.logger.error(f"Error in training_progress: {str(e)}")
        # For AJAX requests, return JSON error
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'error': str(e),
                'logs': {
                    'current_round': 0,
                    'current_epoch': 0,
                    'validation_accuracy': 0.0,
                    'training_complete': False,
                    'status_message': f'Error: {str(e)}',
                    'history': []
                },
                'config': {
                    "num_rounds": 0,
                    "epochs": 0,
                    "batch_size": 0,
                    "clients": []
                }
            }), 500
        
        # For regular requests, return error page
        return str(e), 400


if __name__ == '__main__':
    app.run(debug=True) 