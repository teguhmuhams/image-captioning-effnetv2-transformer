import torch
from model import ImageCaptioningModel
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from dataset import get_transforms

def inference(image, loaded_path, d_model, n_head, vocab, cnn_backbone, transformed = False, device = 'cuda'):
    fc_dim = 2048
    num_layers = 1
    max_seq_length = 20
    image_size = 416
    dropout = 0.1
    transformed_image = image
        
    transforms_pipeline = get_transforms(image_size)
    if not transformed:
        transformed_image = transforms_pipeline(image)
        transformed_image = transformed_image.unsqueeze(dim=0).to(device)

    model = ImageCaptioningModel(cnn_backbone, image_size, len(vocab), max_seq_length, num_layers, n_head, d_model, fc_dim, dropout).to(device)
    load = torch.load(loaded_path, map_location=device)
    model.load_state_dict(load['model_state_dict'])
    model.eval()
    
    return greedy_inference(transformed_image, model, vocab, max_seq_length)
    # return beam_search_inference(transformed_image, model, vocab, max_seq_length, beam_width=5)


def beam_search_inference(transformed_image, model, vocab, max_seq_length, beam_width=5, device='cuda'):
    # Initialize the beam with the start token
    beams = [(0.0, [vocab["<start>"]])]  # (cumulative log prob, seq)
    
    # Move image to device once
    transformed_image = transformed_image.to(device)

    for _ in range(max_seq_length - 1):
        # Prepare to batch process beams
        all_candidates = []
        for log_prob, seq in beams:
            if seq[-1] == vocab["<end>"]:
                all_candidates.append((log_prob, seq))
                continue

            # Prepare input tensor
            caption = torch.LongTensor(seq).unsqueeze(0).to(device)
            mask = torch.zeros((1, len(seq)), dtype=torch.bool).to(device)

            # Predict the next word probabilities
            with torch.no_grad():
                pred, _ = model(transformed_image, caption, mask)
            pred = torch.softmax(pred[:, -1], dim=-1)

            # Get top beam_width predictions
            top_preds = torch.topk(pred, beam_width)
            values = top_preds.values.squeeze()
            indices = top_preds.indices.squeeze()

            # Ensure values and indices are iterable
            if values.dim() == 0:
                values = values.unsqueeze(0)
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)

            # Create new candidates
            for prob, idx in zip(values, indices):
                new_seq = seq + [idx.item()]
                new_prob = log_prob + prob.log().item()  # Update log probability
                all_candidates.append((new_prob, new_seq))

        # Prune beams to top `beam_width`
        beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]

    # Choose the best beam after finishing
    best_beam = max(beams, key=lambda x: x[0])
    target_tokens = [vocab.get_itos()[i] for i in best_beam[1]]

    return target_tokens[1:]

def greedy_inference(transformed_image, model, vocab, max_seq_length, device='cuda'):
    # Initialize the caption with the start token
    caption = [vocab["<start>"]]
    
    # Move image to device once
    transformed_image = transformed_image.to(device)

    for _ in range(max_seq_length - 1):
        # Prepare input tensor
        caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(device)
        mask = torch.zeros((1, len(caption)), dtype=torch.bool).to(device)

        # Predict the next word probabilities
        with torch.no_grad():
            pred, _ = model(transformed_image, caption_tensor, mask)
        pred = torch.softmax(pred[:, -1], dim=-1)

        # Get the most probable next word
        next_word = pred.argmax(dim=-1).item()
        caption.append(next_word)

        # Stop if the end token is generated
        if next_word == vocab["<end>"]:
            break

    # Convert token ids to words
    target_tokens = [vocab.get_itos()[i] for i in caption]

    return target_tokens[1:]  # Exclude the start token