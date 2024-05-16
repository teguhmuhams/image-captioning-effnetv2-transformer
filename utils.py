import os, re
import torch
import matplotlib.pyplot as plt
import pandas as pd

def save_checkpoint(dir, epoch, model_state, optimizer_state, train_loss, val_loss, epoch_duration):
    # Save checkpoint
    data = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state': optimizer_state,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch_duration': epoch_duration
    }
    checkpoint_filename = f"{dir}/checkpoint_epoch_{epoch+1}.pth"
    torch.save(data, checkpoint_filename)
    print(f"Checkpoint saved for epoch {epoch+1}")
    
def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    return checkpoint['epoch'], checkpoint['model_state_dict'], checkpoint['optimizer_state'], checkpoint['train_loss'], checkpoint['val_loss'], checkpoint['epoch_duration']

# After training completes
def save_model(model, path='model.pth'):
    torch.save(model, path)

def safe_parse_int(s):
    try:
        return int(s.split('_')[1].split('.')[0])
    except ValueError:
        return float('inf')  # or return -1 if you prefer to sort these at the start

def natural_keys(text):
    """ 
    A helper function that returns a list of elements in the string as integers and non-integers,
    which is used for natural human sorting.
    """
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def plot_losses(directory, title=''):
    # Initialize lists to store the aggregated data
    data = []

    # List all files in the directory and sort them
    checkpoint_files = sorted(os.listdir(directory), key=natural_keys)
        
    # Load each checkpoint and append the data
    for filename in checkpoint_files:
        filepath = os.path.join(directory, filename)
        epoch, _, _, train_loss, val_loss, epoch_duration = load_checkpoint(filepath)
        data.append({
            'Epoch': epoch,
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
            'Epoch Duration': epoch_duration
        })

    # Create DataFrame from the collected data
    df = pd.DataFrame(data, index=None)

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Train Loss'], label='Training Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    
    # Adding titles and labels
    plt.title(f'Training and Validation Loss - {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Show the plot
    plt.show()

    # Return the DataFrame
    return df