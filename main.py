import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
import os
from dataset import clean_data, create_text_data, split_data, get_transforms
import torch
from dataset import Flickr8kDataset
from data_loader import get_loader
from model import ImageCaptioningModel
import time
import torch.nn as nn
import torch.optim as optim
from utils import save_model, save_checkpoint, find_latest_checkpoint, load_checkpoint
from train import train, validate
from evaluate import evaluate


if __name__ == '__main__':
    
    # init directory and file path
    root_path = '../'
    data_root_path = f"{root_path}dataset"
    text_dataset = f"{data_root_path}/text/Flickr8k.token.bahasa.txt"
    clean_text_dataset = f"{data_root_path}/text/Flickr8k.token.bahasa.clean.txt"

    train_file = data_root_path + "/text/" + "Flickr_8k.trainImages.txt"
    val_file = data_root_path +  "/text/" + "Flickr_8k.devImages.txt"
    test_file = data_root_path + "/text/" + "Flickr_8k.testImages.txt"
    
    # init tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Clean dataset
    if not os.path.exists(clean_text_dataset):
        clean_data(text_dataset, clean_text_dataset, data_root_path)
        
    dataframe, vocab = create_text_data(clean_text_dataset, tokenizer=tokenizer)

    # split data train, val, test
    train_df, val_df, test_df = split_data(dataframe, train_file, val_file, test_file)

    # Hyperparameters
    d_model = 512
    fc_dim = 2048
    vocab_size = len(vocab)
    num_layers = 1
    n_head = 4
    max_seq_len = 20
    image_size = {
        's': {
            'train': 288,
            'val_test': 384
        },
        'm': {
            'train': 320,
            'val_test': 416
        }
    }
    num_epochs = 25
    dropout = 0.1
    batch_size = 32
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cnn_size = 'm'
        
    # Prepare transforms
    cnn_backbone = {
        's': 'efficientnetv2_rw_s.ra2_in1k',
        'm': 'efficientnetv2_rw_m.agc_in1k', 
    }
    train_transforms = get_transforms(image_size[cnn_size]['train'])
    val_test_transforms = get_transforms(image_size[cnn_size]['val_test'])

    # Init dataset
    train_dataset = Flickr8kDataset(data_root_path, train_df, vocab, tokenizer, max_seq_len, train_transforms)
    val_dataset = Flickr8kDataset(data_root_path, val_df, vocab, tokenizer, max_seq_len, val_test_transforms)
    test_dataset = Flickr8kDataset(data_root_path, test_df, vocab, tokenizer, max_seq_len, val_test_transforms)

    # Init loader
    train_loader, val_loader, test_loader = get_loader(train_dataset, val_dataset, test_dataset, batch_size, num_workers)

    training = False
    test = True
    
    model = ImageCaptioningModel(
        cnn_backbone=cnn_backbone[cnn_size],
        image_size=image_size[cnn_size]['train'],
        d_model=d_model,
        fc_dim=fc_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        n_head=n_head,
        max_seq_length=max_seq_len,
        dropout=dropout
    ).to(device)
    
    if (training):    
        # Training and Fine-tuning loop

        # For optimizer
        learning_rate = 0.0001

        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize or load checkpoint
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3
        best_epoch = 0
        best_model = model
        
        # Set Checkpoint
        checkpoint_directory = f'checkpoints/{d_model}_{n_head}_{cnn_size}'
        
        if not os.path.exists(checkpoint_directory):
            # Create the directory
            os.makedirs(checkpoint_directory)
        latest_checkpoint = find_latest_checkpoint(checkpoint_directory)

        if latest_checkpoint:
            start_epoch, model_state, optimizer_state, train_loss, val_loss, epoch_duration = load_checkpoint(latest_checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            best_val_loss = val_loss  # Consider what's the best strategy for your case
            print(f"Resuming from epoch {start_epoch + 1} with validation loss of {val_loss:.4f}")

        #  Count full training time taken
        training_duration = time.time()
        
        # Main training loop
        for epoch in range(start_epoch, num_epochs):
            step = epoch * len(train_loader)
            # update_learning_rate(optimizer, step, warmup_steps, learning_rate, final_lr)
            
            start_time = time.time()
            train_loss = train(model, train_loader, optimizer, criterion, device, vocab_size, epoch)
            val_loss = validate(model, val_loader, criterion, device, vocab_size, epoch)
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} sec')
            
            save_checkpoint(checkpoint_directory, epoch, model.state_dict(), optimizer.state_dict(), train_loss, val_loss, epoch_duration)
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch+1
                best_model = model
                print(f"Improved validation loss at epoch {epoch+1}")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss for {patience_counter} epochs.")
            
            if patience_counter >= patience:
                print(f"Early stopping... Best validation loss at {best_epoch} epoch.")
                break
        
        # End training time
        training_duration = time.time() - training_duration
        
        # Set model name based on d_model and n_head
        model_save_path = f'saved_model/{d_model}_{n_head}_{cnn_size}.pth'

        model_state = {
            'epoch': epoch, 
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_duration': training_duration,
            'val_loss': best_val_loss
        }
        save_model(model_state, model_save_path)
        print(f'Model saved successfully at {model_save_path}')
                
    if(test):
        evaluate(data_root_path=data_root_path, test_df = test_df, model=model)
        