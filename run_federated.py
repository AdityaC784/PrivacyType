import argparse
import os
import sys
import subprocess
import json
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Federated Learning with dynamic client configuration')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of clients for federated learning')
    parser.add_argument('--num_rounds', type=int, default=4, help='Number of federated learning rounds')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per round')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--validation_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--max_sequence_length', type=int, default=8, help='Maximum sequence length')
    parser.add_argument('--min_word_frequency', type=int, default=2, help='Minimum word frequency')
    return parser.parse_args()

def update_training_logs(round_num=None, epoch=None, val_accuracy=None, message=None, complete=None):
    """Update the training logs file with current progress."""
    try:
        with open('training_logs.json', 'r') as f:
            logs = json.load(f)
        
        if round_num is not None:
            logs['current_round'] = round_num
        if epoch is not None:
            logs['current_epoch'] = epoch
        if val_accuracy is not None:
            logs['validation_accuracy'] = val_accuracy
        if message is not None:
            logs['status_message'] = message
        if complete is not None:
            logs['training_complete'] = complete
            
        with open('training_logs.json', 'w') as f:
            json.dump(logs, f)
            
    except Exception as e:
        print(f"Error updating training logs: {e}")

def check_client_data():
    """Check available client data files."""
    client_files = [f for f in os.listdir('client_data') if f.endswith('.csv')]
    return client_files

def main():
    args = parse_arguments()
    
    # Check available client data
    client_files = check_client_data()
    if not client_files:
        update_training_logs(message="No client data found. Please add data files.", complete=True)
        print("Error: No client data files found in client_data directory.")
        return 1

    # Adjust client count if there are fewer files than requested clients
    if len(client_files) < args.num_clients:
        original_num_clients = args.num_clients
        args.num_clients = len(client_files)
        update_training_logs(message=f"Adjusted client count from {original_num_clients} to {args.num_clients} based on available data.")
        print(f"Warning: Adjusted client count from {original_num_clients} to {args.num_clients} based on available data.")
    
    print("\n===== Federated Learning Configuration =====")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Epochs per round: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Validation split: {args.validation_split}")
    print(f"Max sequence length: {args.max_sequence_length}")
    print(f"Min word frequency: {args.min_word_frequency}")
    print("===========================================\n")
    
    update_training_logs(message=f"Starting training with {args.num_clients} clients, {args.num_rounds} rounds...")
    
    cmd = [
        sys.executable, "main.py",
        "--num_clients", str(args.num_clients),
        "--num_rounds", str(args.num_rounds),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--validation_split", str(args.validation_split),
        "--max_sequence_length", str(args.max_sequence_length),
        "--min_word_frequency", str(args.min_word_frequency)
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Monitor the process and update logs
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                print(line.strip())
                
                # Extract progress information from output
                if "==== Starting Round" in line:
                    try:
                        round_num = int(line.split("Round")[1].strip().split()[0])
                        update_training_logs(round_num=round_num, message=f"Starting round {round_num}...")
                    except:
                        pass
                        
                elif "Epoch" in line and "/" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "Epoch" in part and i < len(parts) - 1:
                                epoch_info = parts[i+1]
                                if "/" in epoch_info:
                                    current_epoch = int(epoch_info.split("/")[0])
                                    update_training_logs(epoch=current_epoch)
                                break
                    except:
                        pass
                
                elif "val_accuracy" in line:
                    try:
                        val_acc_part = line.split("val_accuracy:")[1].strip().split()[0]
                        val_accuracy = float(val_acc_part)
                        update_training_logs(val_accuracy=val_accuracy)
                    except:
                        pass
            
            time.sleep(0.1)
            
        # Process completed
        return_code = process.wait()
        
        if return_code == 0:
            update_training_logs(message="Training completed successfully!", complete=True)
            print("Federated learning completed successfully!")
        else:
            update_training_logs(message=f"Training failed with exit code {return_code}", complete=True)
            print(f"Error: Training failed with exit code {return_code}")
            return return_code
        
    except subprocess.CalledProcessError as e:
        update_training_logs(message=f"Error running federated learning: {e}", complete=True)
        print(f"Error running federated learning: {e}")
        return 1
    except Exception as e:
        update_training_logs(message=f"Unexpected error: {e}", complete=True)
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
