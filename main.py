# imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time
import torch
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import torchvision.models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
from torchvision import datasets, transforms
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(0)

# PART 1: WikiArt -------------------------------------------------------------

# wikiart directory
os.chdir('D:/Wikiart')
# contruct dataframe from csv file
df_wikiart = pd.read_csv('classes.csv', encoding="latin-1")
# only keep file path and artist
df_wikiart = df_wikiart[['filename', 'artist']]
# add root path to filename
df_wikiart['filename'] = 'D:/Wikiart/Images/' + df_wikiart['filename']
# get rid of final rows which contained "unknown" artist
df_wikiart = df_wikiart[0:79998]
# rename columns
df_wikiart = df_wikiart.rename(columns={"filename": "path"})
# capitalize artist names
df_wikiart['artist'] = df_wikiart['artist'].apply(lambda x: ' '.join([i.capitalize() for i in x.split()]))

df = df_wikiart

# filter dataframe into only top 10 artists
artist_counts = df['artist'].value_counts()
top_10_artists = artist_counts.nlargest(10).index
mask = df['artist'].isin(top_10_artists)
df = df[mask]
df = df.reset_index(drop=True)

'''
# Group the dataframe by artist and count the number of paintings per artist
artist_counts = df.groupby('artist')['path'].count()

# Get the top 10 artists with the most paintings
top_10_artists = artist_counts.sort_values(ascending=False)

# Create a bar chart/histogram with a different color bar for each of the top 10 artists
plt.bar(top_10_artists.index, top_10_artists.values, color='dodgerblue')
plt.title('Top 10 Artists with the Most Paintings')
plt.xlabel('Artist')
plt.ylabel('Number of Paintings')
plt.xticks(rotation=90)
plt.show()

# count the number of occurrences of each unique name in the 'artists' column
counts = df['artist'].value_counts()

# print the counts
print(counts)
'''

# undersample and oversample to reach 765 images per class
grouped = df.groupby('artist').agg({'path': 'count'})
over_765 = grouped[grouped['path'] >= 765].index
# under_1000 = grouped[grouped['path'] < 1000].index
df = pd.concat([df[df['artist']==a].sample(n=765, random_state=42) for a in over_765])
# class_under_1000_oversampled = pd.concat([df[df['artist']==a].sample(n=1000, replace=True, random_state=42) for a in under_1000])
df = df.reset_index(drop=True)

'''
image_counts = df['artist'].value_counts().sort_values(ascending=False)
print(image_counts)
'''

'''
# Group the dataframe by artist and count the number of paintings per artist
artist_counts = df.groupby('artist')['path'].count()

# Get the top 10 artists with the most paintings
top_10_artists = artist_counts.sort_values(ascending=False)

# Create a bar chart/histogram with a different color bar for each of the top 10 artists
plt.bar(top_10_artists.index, top_10_artists.values, color='dodgerblue')
plt.title('Top 10 Artists with the Most Paintings')
plt.xlabel('Artist')
plt.ylabel('Number of Paintings')
plt.xticks(rotation=90)
plt.show()

# count the number of occurrences of each unique name in the 'artists' column
counts = df['artist'].value_counts()

# print the counts
print(counts)
'''

'''
# print 5 sample images from dataframe
fig, axes = plt.subplots(1, 5, figsize=(20,10))

for i in range(5):
  sample = df.sample(n=1)
  path = sample['path'].iloc[0]
  artist = sample['artist'].iloc[0]
  image = Image.open(path)
  axes[i].imshow(image)
  axes[i].set_title("Artist: " + artist)
  axes[i].axis('off')
plt.show()
'''

# make arists and label relationship dictionaries
unique_artists = sorted(df['artist'].unique())
artist_label = {artist: label for label, artist in enumerate(unique_artists)}
label_artist = {value: key for key, value in artist_label.items()}

# add labels to each image in dataframe corresponding to artist
df['label'] = df['artist'].map(artist_label)
df.reset_index(drop=True, inplace=True)

# create list of tuples (image path, artist) from dataframe
image_tuples = []
for i in range(len(df)):
  image_tuples.append([df.loc[i, 'path'], df.loc[i, 'label']])
  
# reset directory
os.chdir('D:/')

# crops each image into a square of maximum size
class SquareCrop(object):
    def __call__(self, img):
        w, h = img.size
        size = min(w, h)
        return transforms.CenterCrop(size)(img)

# applies squarecrop, resizes to 224 x 224, converts tot ensor, and normalizes pixel values
transform = transforms.Compose([
    SquareCrop(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])

# function to convert grayscale images to RGB when loading them
def load_image(file):
    image = Image.open(file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

# custom dataset class that takes in list of paths to images and makes a dataset using those images
class ImageDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path, label = self.image_list[index]
        
        # load the image using defined function
        image = load_image(image_path)
        
        # apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        # return image and label
        return image, label

# create the dataset
dataset = ImageDataset(image_tuples, transform=transform)

# define splits
train_size = math.ceil(0.7*len(dataset))-1
val_size = math.ceil(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

'''
print('Number of items in training set =', train_size)
print('Number of items in validation set =', val_size)
print('Number of items in testing set =', test_size)
'''

# split the dataset
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

#------------------------------------------------------------------------------
# Dataloaders

'''
# random trianing samples
tform = transforms.ToPILImage()

# get 5 random samples from the dataset
indices = np.random.choice(len(train_set), size=5, replace=False)
samples = [train_set[i] for i in indices]

# print 5 sample images from dataset
fig, axes = plt.subplots(1, 5, figsize=(20,10))

for i, (image, label) in enumerate(samples):
    image = tform(image)
    axes[i].imshow(image)
    axes[i].set_title("Artist: " + label_artist[label])
    axes[i].axis('off')
plt.show()
'''

#------------------------------------------------------------------------------
# Subsets

# generate random indices for subset
train_subset_indices = torch.randperm(train_size)[:math.floor(train_size/10)]
val_subset_indices = torch.randperm(val_size)[:math.floor(val_size/10)]
test_subset_indices = torch.randperm(test_size)[:math.floor(test_size/10)]

# create subset of dataset using Subset class
train_subset = Subset(train_set, train_subset_indices)
val_subset = Subset(val_set, val_subset_indices)
test_subset = Subset(test_set, test_subset_indices)

'''
# use DataLoader to load samples from subset
train_subset_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
val_subset_loader = DataLoader(val_subset, batch_size=1, shuffle=True)

training_images = []
training_labels = []

for image, label in train_subset_loader:
  training_images.append(torch.flatten(image).tolist())
  training_labels.append(int(label[0]))

model = KNeighborsClassifier(n_neighbors=10)
model.fit(training_images, training_labels)

correct = 0
correct_labels = []
pred_labels = []
for image, label in val_subset_loader:
    correct_labels.append(label)
    pred_labels.append(model.predict([torch.flatten(image).tolist()]))
    if model.predict([torch.flatten(image).tolist()]) == int(label[0]):
      correct += 1

print(correct / len(val_subset_loader))

for i in pred_labels:
  if len(i) == 2:
    i.pop(1)
    
cm = np.zeros((10, 10))
for i in range(len(correct_labels)):
    cm[correct_labels[i]][pred_labels[i]] += 1

df_confusion_matrix = pd.DataFrame(cm, index = [i for i in artist_label.keys()], columns = [i for i in artist_label.keys()])
plt.figure(figsize = (10, 7))
sns.heatmap(df_confusion_matrix, annot=True)
'''

#------------------------------------------------------------------------------
# Training. Accuracy, Loss, and F1 Funcitons

def get_f1_score(model, dataloader, displayConfusionMatrix=False):
    true_labels = []
    pred_labels = []

    for imgs, labels in dataloader:
        output = model(imgs)
        predicted = torch.max(output.data, 1)[1]
        true_labels += labels.cpu().numpy().tolist()
        pred_labels += predicted.cpu().numpy().tolist()

    # set the 'average' parameter to 'weighted' for multi-class problems
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    if displayConfusionMatrix == True:
        cm = np.zeros((10, 10))
        for i in range(len(true_labels)):
            cm[true_labels[i]][pred_labels[i]] += 1

        df_confusion_matrix = pd.DataFrame(cm, index = [i for i in artist_label.keys()], columns = [i for i in artist_label.keys()])
        plt.figure(figsize = (10, 7))
        sns.heatmap(df_confusion_matrix, annot=True)

        print("The F1 score is:", f1)
        return None

    return f1

def get_accuracy(model, dataloader):
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        output = model(imgs)
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def get_val_loss(model, val_loader, criterion):
    total_loss = 0
    num_samples = 0
    for imgs, labels in iter(val_loader):
        output = model(imgs)
        loss = criterion(output, labels)
        total_loss += loss.item() * len(imgs)
        num_samples += len(imgs)
  
    val_loss = total_loss / num_samples
    return val_loss

def train_model(path, model, train_dataset, val_dataset, batch_size, num_epochs, learning_rate):
    
    #orch.manual_seed(1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    iters, train_loss, val_loss, train_f1, val_f1, train_acc, val_acc = [], [], [], [], [], [], []
    
    # training
    print("Training started...\n")
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0
        num_samples = 0
        start = time.time()
        for imgs, labels in iter(train_loader):
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch  
            
            epoch_loss += loss.item() * len(imgs)
            num_samples += len(imgs)
        
        scheduler.step()      # step for scheduler

        # save the current training information
        iters.append(epoch)

        # compute average train loss for batch
        train_loss.append(float(epoch_loss)/num_samples)
        # compute average validation loss for this batch
        val_loss.append(get_val_loss(model, val_loader, criterion))
        
        # compute training f1 score
        train_f1.append(get_f1_score(model, train_loader))
        # compute validation f1 score
        val_f1.append(get_f1_score(model, val_loader))

        # compute training accuracy
        train_acc.append(get_accuracy(model, train_loader))
        # compute validation accuracy
        val_acc.append(get_accuracy(model, val_loader))

        # save the current model (checkpoint) to a file
        model_name = "{0}v3_batch-size-{1}_epoch-{2}_learn-rate-{3}.pth".format(model.name, batch_size, epoch, learning_rate)
        torch.save(model.state_dict(), path + model_name)
        
        # determine time elapsed
        end = time.time()
        t = end - start

        # print progress
        print('Epoch {} complete in {} seconds with:\n'
          'Training F1 score: {}, Validation F1 score: {}\n' 
          'Training Accuracy: {}, Validation Accuracy: {}\n'
        'Training loss: {}, Validation Loss: {}\n'.format(epoch, round(t, 5), round(train_f1[-1], 5), round(val_f1[-1], 5), round(train_acc[-1], 5), round(val_acc[-1], 5), round(train_loss[-1], 5), round(val_loss[-1], 5)))
        
    # plotting
    plt.title("Loss Curve")
    plt.plot(iters, train_loss, label="Train")
    plt.plot(iters, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.xticks(iters)
    plt.show()
 
    plt.title("F1 Score Curve")
    plt.plot(iters, train_f1, label="Train")
    plt.plot(iters, val_f1, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend(loc='best')
    plt.xticks(iters)
    plt.show()

    plt.title("Accuracy Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.xticks(iters)
    plt.show()

#------------------------------------------------------------------------------
# Model

num_classes = 10
batch_size = 64

resnet18 = models.resnet18(weights='DEFAULT')

# freeze hidden layer parameters
for param in resnet18.parameters():
    param.requires_grad = False
    
resnet18.name='resnet18'

# fine tune classification/fully connected layer weights
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

primary_model = resnet18

# test loaders
test_subset_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

#------------------------------------------------------------------------------

# save models to this path
path = 'D:/Models/'
    
# perform training
train_model(path = path, model = primary_model, train_dataset = train_set, val_dataset = val_set, batch_size=batch_size, num_epochs=10, learning_rate=0.1)

#------------------------------------------------------------------------------
# Testing Primary Model After Training

print('Testing AFTER Training:')
get_f1_score(primary_model, test_loader, displayConfusionMatrix=True)
acc = get_accuracy(primary_model, test_loader)
print('Test accuracy:', acc, '\n')

#------------------------------------------------------------------------------
# Testing Model after Reverting to Default Resnet18
'''
resnet18 = models.resnet18(weights='DEFAULT')

# freeze hidden layer parameters
for param in resnet18.parameters():
    param.requires_grad = False
    
resnet18.name='resnet18'

# fine tune classification/fully connected layer weights
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

primary_model = resnet18

print('Testing after reverting to DEFAULT weights:')
get_f1_score(primary_model, test_loader, displayConfusionMatrix=True)
acc = get_accuracy(primary_model, test_loader)
print('Test accuracy:', acc, '\n')
'''

#------------------------------------------------------------------------------
# Loading Saved Model then Testing

# save models to this path
path = 'D:/Models/'

resnet18 = models.resnet18(weights='DEFAULT')

# freeze hidden layer parameters
for param in resnet18.parameters():
    param.requires_grad = False
    
resnet18.name='resnet18'

# fine tune classification/fully connected layer weights
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

primary_model = resnet18

# define the path and file name for the saved model
model_name = 'resnet18v3_batch-size-64_epoch-10_learn-rate-0.1.pth'

# load the saved model from disk
loaded_state_dict = torch.load(path + model_name)

# update the new model's weights with the saved state dictionary
resnet18.load_state_dict(loaded_state_dict)

primary_model = resnet18

# acc = get_accuracy(primary_model, test_loader)

'''
# testing
print('Testing after loading SAVED weights:')
get_f1_score(primary_model, test_loader, displayConfusionMatrix=True)
acc = get_accuracy(primary_model, test_loader)
print('Test accuracy:', acc, '\n')
'''

'''
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))

axis = 1

for _, ax in enumerate(axes):
    if axis == 1:
        found = False
        for imgs, labels in test_loader:
            if found == True:
                axis += 1
                break
            output = primary_model(imgs)
            predicted = torch.max(output.data, 1)[1]
            for i in range(len(labels)):
                if label_artist[labels[i].item()] == 'Camille Pissarro' and label_artist[predicted[i].item()] == 'Camille Pissarro':
                    ax.imshow(imgs[i].permute(1, 2, 0))
                    ax.set_title('Artist: {}\nPredicted Artist: {}'.format(label_artist[labels[i].item()], label_artist[predicted[i].item()]))
                    ax.axis('off')
                    print('found')
                    found = True
                    break
    elif axis == 2:
        found = False
        for imgs, labels in test_loader:
            if found == True:
                axis += 1
                break
            output = primary_model(imgs)
            predicted = torch.max(output.data, 1)[1]
            for i in range(len(labels)):
                if label_artist[labels[i].item()] == 'Claude Monet' and label_artist[predicted[i].item()] == 'Camille Pissarro':
                    ax.imshow(imgs[i].permute(1, 2, 0))
                    ax.set_title('Artist: {}\nPredicted Artist: {}'.format(label_artist[labels[i].item()], label_artist[predicted[i].item()]))
                    ax.axis('off')
                    print('found')
                    found = True
                    break
    elif axis == 3:
        found = False
        for imgs, labels in test_loader:
            if found == True:
                axis += 1
                break
            output = primary_model(imgs)
            predicted = torch.max(output.data, 1)[1]
            for i in range(len(labels)):
                if label_artist[labels[i].item()] == 'Pierre Auguste Renoir' and label_artist[predicted[i].item()] == 'Camille Pissarro':
                    ax.imshow(imgs[i].permute(1, 2, 0))
                    ax.set_title('Artist: {}\nPredicted Artist: {}'.format(label_artist[labels[i].item()], label_artist[predicted[i].item()]))
                    ax.axis('off')
                    print('found')
                    found = True
                    break
    elif axis == 4:
        found = False
        for imgs, labels in test_loader:
            if found == True:
                axis += 1
                break
            output = primary_model(imgs)
            predicted = torch.max(output.data, 1)[1]
            for i in range(len(labels)):
                if label_artist[labels[i].item()] == 'Vincent Van Gogh' and label_artist[predicted[i].item()] == 'Camille Pissarro':
                    ax.imshow(imgs[i].permute(1, 2, 0))
                    ax.set_title('Artist: {}\nPredicted Artist: {}'.format(label_artist[labels[i].item()], label_artist[predicted[i].item()]))
                    ax.axis('off')
                    print('found')
                    found = True
                    break
    else:
        break
'''    

'''
# DEMONSTRATION
lst = ['John Singer Sargent']

for artist in lst: #artist_label.keys():
    print('Generating sample predictions for paintings from {}...'.format(artist))
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
    
    for _, ax in enumerate(axes):
        found = False
        for imgs, labels in test_loader:
            if found == True:
                break
            output = primary_model(imgs)
            predicted = torch.max(output.data, 1)[1]
            for i in range(len(labels)):
                if label_artist[labels[i].item()] == artist and label_artist[predicted[i].item()] == artist:
                    ax.imshow(imgs[i].permute(1, 2, 0))
                    ax.set_title('Artist: {}\nPredicted Artist: {}'.format(label_artist[labels[i].item()], label_artist[predicted[i].item()]))
                    ax.axis('off')
                    found = True
                    break
    plt.show()
'''

'''
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

for _, ax in enumerate(axes):
    for imgs, labels in test_loader:
        output = primary_model(imgs)
        ax.imshow(imgs[0].permute(1, 2, 0))
        # ax.set_title('Artist: {}\nPredicted Artist: {}'.format(label_artist[labels[i].item()], label_artist[predicted[i].item()]))
        ax.axis('off')
        break
'''
          







