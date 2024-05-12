#
# Copyright (c) 2018 Intel Corporation
# Portions Copyright (C) 2019-2023 Maxim Integrated Products, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
CIFAR-100 Dataset
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import urllib.request
import zipfile
import sklearn.datasets
import ai8x

class TimeOfFlightDataset(Dataset):

    # Initializer with optional data directory and transform parameters
    def __init__(self, data_dir, transform=None):
        
        # Store the data directory
        self.data_dir = data_dir

        # Store the transform to be applied
        self.transform = transform

        # Download and extract dataset if not already done
        self.dataset= self.download_and_extract()

        self.process_data()

    # Method to download and extract the dataset
    def download_and_extract(self):
        
        # URL
        url = "https://www.dropbox.com/s/4txj0ob6ovy9jbr/time-of-flight.zip?dl=1"

        # Open the URL and read the data
        u = urllib.request.urlopen(url)
        data = u.read()
        u.close()

        # Write the data to a zip file in the specified directory
        with open("./data/time-of-flight.zip", "wb") as f :
            f.write(data)

        # Extract the zip file
        with zipfile.ZipFile("./data/time-of-flight.zip","r") as zip_ref:
            zip_ref.extractall(".")

        # Delete any .DS_Store files created by macOS
        for root, dirs, files in os.walk('./data/time-of-flight/'):
            for file in files:
                if file.endswith('.DS_Store'):
                    path = os.path.join(root, file)
                    print("Deleting: %s" % (path))
                    if os.remove(path):
                        print("Unable to delete!")
                    else:
                        print("Deleted...")
        
        # Load the dataset using sklearn's load_files method, ensuring it's shuffled
        dataset = sklearn.datasets.load_files('./data/time-of-flight', shuffle='True', encoding='utf-8')

        return dataset
    
    # Method to process the loaded data
    def process_data(self):
        # Process text data
        for index, data in enumerate(self.dataset.data):
            data = data.split('\n')     # Splits each line
            data = data[1].split(',')   # Splits by comas
            self.dataset.data[index] = [int(i) for i in data]   # Converts to integers

        # Convert data to NumPy arrays and normalize
        self.X = np.array(self.dataset.data, dtype='float32') / 255     # Convert the list of data points to a NumPy array and normalize the values
        self.y = np.array(self.dataset.target)                          # Convert the target labels to a NumPy array

        print("dataset processed")

    # Method to return the size of the dataset
    def __len__(self):
        return len(self.dataset.data)

    # Method to get a specific item from the dataset
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.X[idx]                # Get the sample data at the specified index
        sample = sample.reshape( 8, 8,1)    # Reshape the sample for processing
        if self.transform:
            sample = self.transform(sample) # Apply any specified transforms

        # Return the transformed sample and its label
        return sample, self.y[idx]


# Function to load the dataset, optionally splitting it into training and test sets
def TOF_get_datasets(data, load_train=True, load_test=True):
    """
    Load the CIFAR100 dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-128/128, 127/128]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    
    # Assuming 'data_dir' and 'args' are defined
    (data_dir, args) = data

    # Define the transformations to be applied to the training data
    train_transform = transforms.Compose([
    transforms.ToTensor(),  # Assuming the custom dataset returns PIL images or NumPy arrays
    ai8x.normalize(args=args)
    ])

     # Initialize the TimeOfFlightDataset with the directory and transformations
    tof_dataset = TimeOfFlightDataset(data_dir=data_dir, transform=train_transform)
    print("loading dataet finished",tof_dataset)

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(tof_dataset))
    test_size = len(tof_dataset) - train_size
    train_dataset, test_dataset = random_split(tof_dataset, [train_size, test_size])

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'TOF',
        'input': (1, 8, 8),
        'output': ("background","robot"),
        'loader': TOF_get_datasets,
    },
]
