import pandas as pd
from collections import Counter
import torchtext.vocab; torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from torchvision import transforms

def get_transforms(size):
    pipeline = transforms.Compose([
        transforms.Resize(size=(size,size), antialias=True),
        transforms.CenterCrop(size=(size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # if is_training:
    #     pipeline = transforms.Compose([
    #         transforms.RandomResizedCrop(size=(288, 288), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    return pipeline


def clean_data(raw_file_path, output_file_path, data_root_path):
    with open(raw_file_path, 'r', encoding='utf-8') as file, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in file:
            image_filename, caption = line.split("\t")
            image_filename = image_filename.split('.')[0] + ".jpg"
            
            # remove line if image filename doesnt exists
            if not os.path.exists(os.path.join(data_root_path, "images/", image_filename)):
                continue
            
            outfile.write(f"{image_filename}\t{caption}")
  

def split_data(df, train_file, val_file, test_file):
    with open(train_file, 'r') as file:
        train_filenames = file.read().strip().split('\n')
    
    with open(val_file, 'r') as file:
        val_filenames = file.read().strip().split('\n')
    
    with open(test_file, 'r') as file:
        test_filenames = file.read().strip().split('\n')
    
    df_train = df[df['image_filename'].isin(train_filenames)]
    df_val = df[df['image_filename'].isin(val_filenames)]
    df_test = df[df['image_filename'].isin(test_filenames)]
    
    return df_train, df_val, df_test


def create_text_data(filename, tokenizer):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        image2caption_mapping = {}
        text_data = []

    for line in caption_data:
        line = line.rstrip("\n")
        image_filename, caption = line.split("\t")

        text_data.append(caption)

        # Create a dictionary that maps an image filename to a list of captions.
        if image_filename in image2caption_mapping:
            image2caption_mapping[image_filename].append(caption)
        else:
            image2caption_mapping[image_filename] = [caption]

    # Create a dataframe that contains image filename and the respective captions
    dataframe = pd.DataFrame(columns=["image_filename", "captions"])
    dataframe["image_filename"] = image2caption_mapping.keys()
    dataframe["captions"] = dataframe["image_filename"].map(
        lambda x: image2caption_mapping[x]
    )

    # Create torchtext vocab
    counter = Counter()
    for caption in text_data:
        counter.update(tokenizer(caption))

    vocab = torchtext.vocab.vocab(counter)
    for special_token, index in zip(["<unk>", "<pad>", "<start>", "<end>"], [0, 1, 2, 3]):
        vocab.insert_token(special_token, index)
    vocab.set_default_index(0)
    
    return dataframe, vocab

class Flickr8kDataset(Dataset):
    def __init__(
        self, data_root_path, dataframe, vocab, tokenizer, max_seq_len, transforms
    ):
        self.data_root_path = data_root_path
        self.dataframe = dataframe.copy()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.transforms = transforms

    def __getitem__(self, idx):
        # Image
        image_filename, captions = self.dataframe.iloc[idx]
        image = Image.open(f"{self.data_root_path}/images/{image_filename}").convert('RGB')
        image = self.transforms(image)
        
        # Captions
        tokens_arrays = []
        attention_masks = []
        pad_idx = self.vocab["<pad>"]
        start_idx = self.vocab["<start>"]
        end_idx = self.vocab["<end>"]
        pad_starts = None

        for i, caption in enumerate(captions):
            # Add start and end tokens
            tokens = [self.vocab[token] for token in self.tokenizer(caption)]
            tokens = [start_idx] + tokens + [end_idx]
            
            # Pad tokens and ensure the length equals self.max_seq_len
            if len(tokens) <= self.max_seq_len:
                pad_starts = len(tokens)
                tokens += [pad_idx] * (self.max_seq_len - len(tokens))
            else:
                tokens = tokens[: self.max_seq_len - 1] + [end_idx]

            assert len(tokens) == self.max_seq_len  # TODO: move to test
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens_arrays.append(tokens)

            # create 
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            if pad_starts is not None:
                mask[pad_starts:] = True
            attention_masks.append(mask)

        out_dict = {
            "image": image,
            "image_filename": image_filename,
            "captions_tokens": tokens_arrays,
            "captions": captions,
            "attention_masks": attention_masks,
        }
        
        return out_dict

    def __len__(self):
        return len(self.dataframe)

    def indices_to_tokens(self, indices, index=None):
        itos = self.vocab.get_itos()

        if index is not None:
            if index < len(indices):
                indices_tensor = indices[index]
                tokens = [itos[ind] for ind in indices_tensor.tolist() if ind < len(itos)]
                return tokens
            else:
                raise IndexError("Index out of range")
            

if __name__ == '__main__':
    print("hello")