from tqdm import tqdm
import torch

def train(model, data_loader, optimizer, criterion, device, vocab_size, epoch):
    model.train()
    total_loss = 0.0
    train_iterator = tqdm(data_loader, desc='Training', leave=False, total=len(data_loader))
            
    for data in train_iterator:
        image = data['image'].to(device)
        captions = data['captions_tokens']
        attention_masks = data['attention_masks']
        
        optimizer.zero_grad()
        loss = 0.0
        
        for caption, mask in zip(captions, attention_masks):
            caption_input = caption[:, :-1].to(device)
            mask_input = mask[:, :-1].to(device)

            pred, _ = model(image, caption_input, mask_input)
            pred = pred.view(-1, vocab_size)

            caption_target = caption[:, 1:].contiguous().view(-1).to(device)
            loss += criterion(pred, caption_target)

        loss /= len(captions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_iterator.set_postfix({'epoch': epoch + 1, 'loss': loss.item()})

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def validate(model, data_loader, criterion, device, vocab_size, epoch):
    model.eval()
    total_loss = 0.0
    
    val_iterator = tqdm(data_loader, desc='Validating', leave=False, total=len(data_loader))
    with torch.no_grad():
        for data in val_iterator:
            image = data['image'].to(device)
            captions = data['captions_tokens']
            attention_masks = data['attention_masks']

            loss = 0.0
            num_captions_per_image = len(captions[0])
            for caption, mask in zip(captions, attention_masks):
                caption_input = caption[:, :-1].to(device)
                mask_input = mask[:, :-1].to(device)

                pred, _ = model(image, caption_input, mask_input)
                pred = pred.view(-1, vocab_size)

                caption_target = caption[:, 1:].contiguous().view(-1).to(device)
                loss += criterion(pred, caption_target)
            
            loss /= num_captions_per_image
            total_loss += loss.item()
            val_iterator.set_postfix({'epoch': epoch + 1, 'val_loss': loss.item()})

    avg_loss = total_loss / len(data_loader)
    return avg_loss
