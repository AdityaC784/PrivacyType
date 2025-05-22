import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.ticker import MaxNLocator

def plot_training_history(history_file, output_dir='visualizations'):
    """Plot training history from a JSON file."""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group logs by round and client
    logs_by_round_client = {}
    for log in history:
        round_num = log['round']
        client_id = log['client_id']
        key = (round_num, client_id)
        
        if key not in logs_by_round_client:
            logs_by_round_client[key] = []
        
        logs_by_round_client[key].append(log)
    
    # Get unique rounds and clients
    rounds = sorted(set(key[0] for key in logs_by_round_client.keys()))
    clients = sorted(set(key[1] for key in logs_by_round_client.keys()))
    
    # Plot metrics for each client by round
    metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    colors = plt.cm.tab10(np.linspace(0, 1, len(clients)))
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for i, client_id in enumerate(clients):
            client_rounds = []
            client_metrics = []
            
            for round_num in rounds:
                key = (round_num, client_id)
                if key in logs_by_round_client:
                    client_logs = logs_by_round_client[key]
                    last_epoch_metric = client_logs[-1].get(metric, 0)
                    client_rounds.append(round_num)
                    client_metrics.append(last_epoch_metric)
            
            if client_metrics:
                plt.plot(client_rounds, client_metrics, 'o-', label=f'Client {client_id}', color=colors[i])
        
        plt.title(f'{metric.capitalize()} by Round')
        plt.xlabel('Round')
        plt.ylabel(metric.capitalize())
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{metric}_by_round.png'))
        plt.close()
    
    # Plot average metrics across all clients by round
    plt.figure(figsize=(12, 6))
    
    for metric in ['loss', 'accuracy']:
        avg_metrics = []
        
        for round_num in rounds:
            round_metrics = []
            for client_id in clients:
                key = (round_num, client_id)
                if key in logs_by_round_client:
                    client_logs = logs_by_round_client[key]
                    last_epoch_metric = client_logs[-1].get(metric, 0)
                    round_metrics.append(last_epoch_metric)
            
            if round_metrics:
                avg_metrics.append(np.mean(round_metrics))
            else:
                avg_metrics.append(0)
        
        plt.plot(rounds, avg_metrics, 'o-', label=f'Average {metric.capitalize()}')
    
    plt.title('Model Performance by Round')
    plt.xlabel('Round')
    plt.ylabel('Metric Value')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'performance_by_round.png'))
    plt.close()
    
    # Plot metrics by epoch for each round and client
    for metric in metrics:
        plt.figure(figsize=(15, 10))
        
        for i, (round_num, client_id) in enumerate(logs_by_round_client.keys()):
            logs = logs_by_round_client[(round_num, client_id)]
            epochs = [log['epoch'] for log in logs]
            values = [log.get(metric, 0) for log in logs]
            
            client_index = clients.index(client_id)
            client_color = colors[client_index]
            line_style = ['solid', 'dashed', 'dotted', 'dashdot'][round_num % 4]
            
            plt.plot(epochs, values, 
                     label=f'Round {round_num}, Client {client_id}',
                     color=client_color,
                     linestyle=line_style)
        
        plt.title(f'{metric.capitalize()} by Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{metric}_by_epoch.png'))
        plt.close()
    
    return True

def main():
    """Main function for running the visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training history')
    parser.add_argument('--history_file', type=str, default='training_logs.json', help='Path to the training history JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save the visualizations')
    
    args = parser.parse_args()
    
    plot_training_history(args.history_file, args.output_dir)
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()