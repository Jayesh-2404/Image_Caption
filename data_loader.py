#imports 
"""
data_loader.py

This module provides functionality for loading and processing image-caption datasets for machine learning tasks. 
It includes a custom PyTorch Dataset class, a custom collate function for batching, and DataLoader setup for 
training and testing.

Classes:
    ImageCaptionDataset:
        A PyTorch Dataset class for loading images and their corresponding captions. 
        Captions are tokenized using a specified tokenizer.

Functions:
    custom_collate_fn(batch):
        A custom collate function to handle batching of images and captions. 
        Pads captions to the same length and stacks images into a batch.

Usage:
    - The `ImageCaptionDataset` class is used to load image-caption pairs from a specified directory and captions file.
    - The `custom_collate_fn` function is used to prepare batches of data for training/testing.
    - DataLoaders are created for training and testing subsets of the dataset.

Dependencies:
    - pandas
    - os
    - torch
    - torchvision.transforms
    - matplotlib.pyplot
    - numpy
    - PIL.Image
    - sklearn.model_selection
    - transformers.BertTokenizer

Example:
    # Initialize tokenizer and transformations

    # Create dataset and subsets
    dataset = ImageCaptionDataset(root_dir, captions_file, tokenizer, transform)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
"""
import pandas as pd
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transform 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, tokenizer, transform=None):
        self.root_dir = root_dir
        self.captions_file = pd.read_csv(captions_file)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.captions_file)

    def __getitem__(self, idx):
        img_name = self.captions_file.iloc[idx, 0]
        caption = self.captions_file.iloc[idx, 1]

        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Tokenize the caption
        caption_tokens = self.tokenizer(caption, padding='max_length', max_length=30, truncation=True, return_tensors="pt")
        caption_tensor = caption_tokens['input_ids'].squeeze()  # Remove extra dimension

        return image, caption_tensor

# for padidng 

def custom_collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import BertTokenizer

# using BERT tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# dataset
root_dir = '/kaggle/input/flickr8k/Images'
captions_file = '/kaggle/input/flickr8k/captions.txt'
dataset = ImageCaptionDataset(root_dir=root_dir, captions_file=captions_file, tokenizer=tokenizer, transform=transform)

# a subset of the dataset with 5000 samples
subset_indices = list(range(6000))
subset = Subset(dataset, subset_indices)

train_size = int(0.8 * len(subset_indices))
test_size = len(subset_indices) - train_size

train_indices, test_indices = train_test_split(subset_indices, train_size=train_size, test_size=test_size, random_state=42)

train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)