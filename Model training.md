## Import library


```python
import os
import pydicom
import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm import create_model
from timm.layers import SelectAdaptivePool2d, LayerNorm2d
```

## Set CUDA device


```python
print(torch.cuda.is_available())
torch.cuda.set_device(0)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
device = torch.device('cuda')
```

## Load dataset

- Load csv file containing metadata and outcome for each id
- Create custom Dataset class containing 4 images, labels, soft-label (probability from XGBoost), metadata (age and sex)
- Preload image to save time
- Set transformation for each image separately: Pre_AP, Pre_LAT, Post_AP, Post_LAT
- Training augmentation includes: Random brightness adjustment, Random horizontal flip, Random rotation +/- 5 degrees


```python
#Load data
case_df = pd.read_csv("Placeholder_dataset.csv")
case_df = case_df.sort_values(by="id")
```


```python
case_df.head()
```


```python
#Train-Test split
test_split = int(len(case_df)*0.8)
train_val_df = case_df.iloc[:test_split]
test_df = case_df.iloc[test_split:]
```


```python
train_val_df.head()
```


```python
test_df.head()
```


```python
#Dataset
class WristFractureDataset(Dataset):
    def __init__(self, df, img_dir, mode, transform="train"):
        self.df = df
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform
        self.prob_labels = []
        self.images = []
        self.labels = []
        self.male = []
        self.age = []

        for idx in range(len(df)):
            row = df.iloc[idx]
            patient_id = f"{int(row['id'])}"
            preloaded_images = []

            for view in ["Pre_AP", "Pre_LAT", "Post_AP", "Post_LAT"]:
                img_path = os.path.join(self.img_dir, view, f"{patient_id}.png")
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                preloaded_images.append(img)

            male = row[['male']].values.astype(float)
            age = row[['age_normalize']].values.astype(float)

            prob_labels = row[['prob_outcome']].values.astype(float)
            
            labels = row[["outcome"]].values.astype(float)

            self.images.append(preloaded_images)
            self.prob_labels.append(prob_labels)
            self.labels.append(labels)
            self.male.append(male)
            self.age.append(age)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        labels = torch.tensor(labels, dtype = torch.float32)
        prob_labels = self.prob_labels[idx]
        prob_labels = torch.tensor(prob_labels, dtype = torch.float32)
        male = self.male[idx]
        male = torch.tensor(male, dtype = torch.long)
        age = self.age[idx]
        age = torch.tensor(age, dtype = torch.float32)
        

        if self.transform == "train":
            pre_ap = transform_train_pre_ap(image=images[0])["image"]
            pre_ap = torch.stack([pre_ap.squeeze(0)]*3, dim = 0)
            pre_lat = transform_train_pre_lat(image=images[1])["image"]
            pre_lat = torch.stack([pre_lat.squeeze(0)]*3, dim = 0)
            post_ap = transform_train_post_ap(image=images[2])["image"]
            post_ap = torch.stack([post_ap.squeeze(0)]*3, dim = 0)
            post_lat = transform_train_post_lat(image=images[3])["image"]
            post_lat = torch.stack([post_lat.squeeze(0)]*3, dim = 0)

        if self.transform == "test":
            pre_ap = transform_test_pre_ap(image=images[0])["image"]
            pre_ap = torch.stack([pre_ap.squeeze(0)]*3, dim = 0)
            pre_lat = transform_test_pre_lat(image=images[1])["image"]
            pre_lat = torch.stack([pre_lat.squeeze(0)]*3, dim = 0)
            post_ap = transform_test_post_ap(image=images[2])["image"]
            post_ap = torch.stack([post_ap.squeeze(0)]*3, dim = 0)
            post_lat = transform_test_post_lat(image=images[3])["image"]
            post_lat = torch.stack([post_lat.squeeze(0)]*3, dim = 0)

        return pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age
```


```python
#Mean and std of image
Pre_AP_Mean =
Pre_AP_Std =
Pre_LAT_Mean =
Pre_LAT_Std =
Post_AP_Mean =
Post_AP_Std =
Post_LAT_Mean =
Post_LAT_Std =
```


```python
# Preprocessing and augmentation
transform_train_pre_ap = A.Compose([
    A.Resize(height=224, width=224),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.HorizontalFlip(p = 0.5),
    A.Affine(rotate = (-5, 5), scale = 1.0, p = 0.5, keep_ratio = True),
    A.Normalize(mean=[Pre_AP_Mean], std=[Pre_AP_Std]),
    ToTensorV2()
])

transform_train_pre_lat = A.Compose([
    A.Resize(height=224, width=224),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.HorizontalFlip(p = 0.5),
    A.Affine(rotate = (-5, 5), scale = 1.0, p = 0.5, keep_ratio = True),
    A.Normalize(mean=[Pre_LAT_Mean], std=[Pre_LAT_Std]),
    ToTensorV2()
])

transform_train_post_ap = A.Compose([
    A.Resize(height=224, width=224),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.HorizontalFlip(p = 0.5),
    A.Affine(rotate = (-5, 5), scale = 1.0, p = 0.5, keep_ratio = True),
    A.Normalize(mean=[Post_AP_Mean], std=[Post_AP_Std]),
    ToTensorV2()
])

transform_train_post_lat = A.Compose([
    A.Resize(height=224, width=224),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.HorizontalFlip(p = 0.5),
    A.Affine(rotate = (-5, 5), scale = 1.0, p = 0.5, keep_ratio = True),
    A.Normalize(mean=[Post_LAT_Mean], std=[Post_LAT_Std]),
    ToTensorV2()
])

transform_test_pre_ap = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[Pre_AP_Mean], std=[Pre_AP_Std]),
    ToTensorV2()
])

transform_test_pre_lat = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[Pre_LAT_Mean], std=[Pre_LAT_Std]),
    ToTensorV2()
])

transform_test_post_ap = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[Post_AP_Mean], std=[Post_AP_Std]),
    ToTensorV2()
])

transform_test_post_lat = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[Post_LAT_Mean], std=[Post_LAT_Std]),
    ToTensorV2()
])
```


```python
# Dataset and Data loader
start_time = time.time()
train_val_dataset = WristFractureDataset(train_val_df, "placeholder_folder", mode="train", transform="train")
test_dataset = WristFractureDataset(test_df, "placeholder_folder", mode="test", transform="test")
print(time.time() - start_time)
```

## Utility function

Utility function for checkpoint saving, setting model dropout rate, plot ROC and Calibration curve


```python
def save_checkpoint(epoch, model, optimizer, best_roc_auc, fold, save_dir="placeholder_folder"):
    # Create a directory for the specific fold
    fold_dir = os.path.join(save_dir, f"fold_{fold}")
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # Save the checkpoint in the fold directory
    checkpoint_path = os.path.join(fold_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_roc_auc': best_roc_auc,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
```


```python
def set_dropout(module, p):
    """Recursively set dropout probability for all Dropout layers."""
    for child_name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, child_name, nn.Dropout(p=p))
        elif isinstance(child, nn.Dropout2d):  # In case of 2D dropout
            setattr(module, child_name, nn.Dropout2d(p=p))
        else:
            set_dropout(child, p)
```


```python
def plot_multiple_roc_and_calibration_curves(y_trues, y_preds, labels=None, n_bins=10):
    """
    Plots the ROC and Calibration curves for multiple sets of results.

    Args:
        y_trues: List of arrays with ground truth labels for each result set.
        y_preds: List of arrays with predicted probabilities for each result set.
        labels: List of labels corresponding to each result set. Defaults to generic names.
        n_bins: Number of bins for the calibration curve.
    """
    if labels is None:
        labels = [f"Result {i+1}" for i in range(len(y_trues))]

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curve
    for y_true, y_pred, label in zip(y_trues, y_preds, labels):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        axs[0].plot(fpr, tpr, label=f'{label} AUC = {auc_score:.4f}')
    
    axs[0].plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line
    axs[0].set_xlabel('False Positive Rate (FPR)')
    axs[0].set_ylabel('True Positive Rate (TPR)')
    axs[0].set_title('ROC Curve')
    axs[0].legend(loc='lower right')
    axs[0].grid()

    # Calibration Curve
    for y_true, y_pred, label in zip(y_trues, y_preds, labels):
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
        axs[1].plot(prob_pred, prob_true, marker='o', label=label)

    axs[1].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')  # Perfect calibration line
    axs[1].set_xlabel('Mean Predicted Probability')
    axs[1].set_ylabel('Fraction of Positives')
    axs[1].set_title('Calibration Curve')
    axs[1].legend(loc='best')
    axs[1].grid()

    # Show plots
    plt.tight_layout()
    plt.show()

```

## Define model

Define MultiImage Model using pre-trained CoAtNet backbone from `timm`

**EmbeddingHead**
Create embedding head to replace last fully connected layer from the original CoAtNet in order to pool embedding from 4 images and metadata for prediction

**ClassifierHead**
Using concatenated embedding from 4 images and metadata to output linear prediction

**MultiImageCoatNet**
Main deep elarning model architecture with custom number of CoAtNet backbone, accept multiple images and metadata as input for forward pass

Each image is passed separately to their respective backbone, then embeddings from each image are concatenated together with expanded metadata to pass into ClassifierHead


```python
class EmbeddingHead(nn.Module):
    def __init__(self, embedding_dim):
        super(EmbeddingHead, self).__init__()
        self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=False)  # Global pooling without flattening
        self.norm = LayerNorm2d((embedding_dim,), eps=1e-6)  # LayerNorm2d for spatial data
        self.flatten = nn.Flatten(start_dim=1)   # Flatten to (batch_size, embedding_dim)

    def forward(self, x):
        x = self.global_pool(x)  # Global pooling: (batch_size, embedding_dim, 1, 1)
        x = self.norm(x)  # Apply LayerNorm2d: (batch_size, embedding_dim, 1, 1)
        x = self.flatten(x)  # Flatten to ensure proper shape: (batch_size, embedding_dim)
        return x

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer before fc2
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second fully connected layer

    def forward(self, x):
        x = self.fc1(x)  # Apply first fully connected layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply second fully connected layer
        return x
```


```python
class MultiImageCoatNet(nn.Module):
    def __init__(self, num_images=4, num_classes=1):
        super(MultiImageCoatNet, self).__init__()

        self.num_images = num_images

        # Define a list of MaxxVit models, one for each image
        self.models = nn.ModuleList()
        for _ in range(num_images):
            coatnet_model = self._create_coatnet_model()
            self.models.append(coatnet_model)

        # Embedding dimension
        
        self.sex_embedding = nn.Embedding(2, 128)
        self.age_projection = nn.Linear(1, 128)
        self.embedding_dim = 768 * num_images + 256
        # Final classifier that combines embeddings and metadata
        self.classifier = ClassifierHead(input_dim = self.embedding_dim, hidden_dim = 1024, output_dim = num_classes, dropout_rate = 0.0)

    def _create_coatnet_model(self):
        """
        Initialize a coatnet model and remove its classifier head.
        """
        coatnet_model = create_model("coatnet_0_rw_224", pretrained=True, drop_path_rate = 0.1)  # Replace with the actual MaxxVit class
        coatnet_model.head = EmbeddingHead(768)  # Change the classifier head to embedding head
        set_dropout(coatnet_model, p=0.1)

        for param in coatnet_model.parameters():
                param.requires_grad = True

        return coatnet_model

    def forward(self, images, male, age):
        """
        Forward pass for the multi-image model.
        
        Args:
            images: List of tensors of shape (batch_size, C, H, W), one for each image.
            male: Tensor of shape (batch_size, 1).
            age: Tensor of shape (batch_size, 1)
        
        Returns:
            out: Tensor of shape (batch_size, num_classes).
        """
        assert len(images) == self.num_images, f"Expected {self.num_images} images, got {len(images)}"

        embeddings = []
        for i, model in enumerate(self.models):
            # Pass each image through its corresponding MaxxVit model
            x = model(images[i])  # Shape: (batch_size, embedding_dim)
            embeddings.append(x)

        male_emb = self.sex_embedding(male).squeeze(1)
        embeddings.append(male_emb)
        age_proj = self.age_projection(age.unsqueeze(-1)).squeeze(1)
        embeddings.append(age_proj)
        # Concatenate embeddings along the feature dimension
        concatenated_embeddings = torch.cat(embeddings, dim=1)  # Shape: (batch_size, embedding_dim * num_images)

        # Pass combined features through the classifier
        out = self.classifier(concatenated_embeddings)
        
        return out

```

## Training function

Main model training function, will be used for 4-fold cross validation.

Extract data from dataloader then use soft-label as target for loss calculation and calculate performance using binary label.

After each epoch, output train and validation loss, as well as AUC for validation set and set of other metrics


```python
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50, scheduler=None, fold=1):
    model.to(device)
    best_roc_auc = 0

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []
    metrics_history = []  # To store metrics for each epoch
    
    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        for pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age in train_loader:  
            pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age = (
                pre_ap.to(device), pre_lat.to(device), post_ap.to(device), post_lat.to(device), labels.to(device), prob_labels.to(device), male.to(device), age.to(device)
            )
            
            optimizer.zero_grad()
            outputs = model([pre_ap, pre_lat, post_ap, post_lat], male, age)  # Forward pass with 4 images
            loss = criterion(outputs, prob_labels)         # Calculate loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age in val_loader:  
                pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age = (
                    pre_ap.to(device), pre_lat.to(device), post_ap.to(device), post_lat.to(device), labels.to(device), prob_labels.to(device), male.to(device), age.to(device)
                )
                
                outputs = model([pre_ap, pre_lat, post_ap, post_lat], male, age)  # Forward pass with 4 images
                loss = criterion(outputs, prob_labels)         # Calculate loss
                val_loss += loss.item()

                y_true.extend(labels.cpu().numpy())       # Collect ground truth
                y_pred.extend(torch.sigmoid(outputs).cpu().numpy())  # Collect predictions

        # Average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Calculate metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_binary = (y_pred > 0.5).astype(int)
        roc_auc = roc_auc_score(y_true, y_pred, average="macro")
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)

        metrics_history.append({
            "epoch": epoch + 1,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        # Adjust learning rate with scheduler (if provided)
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
        print(f"  Time: {epoch_time:.2f} seconds")

        # Update best ROC AUC and save checkpoint
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            save_checkpoint(epoch + 1, model, optimizer, best_roc_auc, fold)

                # Save checkpoint every 25 epochs
        elif (epoch + 1) % 25 == 0:
            save_checkpoint(epoch + 1, model, optimizer, best_roc_auc, fold)

    return train_losses, val_losses, best_roc_auc, metrics_history
```

## 4-fold model training

Doing 4-fold cross validation, 200 epochs each


```python
# Parameters
num_folds = 4
batch_size = 64
num_epochs = 200


# Initialize 4-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=False)

# Container for fold results
fold_results = []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_val_dataset)))):
    print(f"Fold {fold + 1}/{num_folds}")

    train_subset = torch.utils.data.Subset(train_val_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_val_dataset, val_idx)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = MultiImageCoatNet(num_images=4, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_losses, val_losses, best_roc_auc, metrics_history = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs, scheduler=scheduler, fold=fold + 1
    )


    # Save results for the current fold
    fold_results.append({
        "fold": fold + 1,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_roc_auc": best_roc_auc,
        "metric_history": metrics_history
    })

# Average performance across folds
avg_roc_auc = np.mean([result["best_roc_auc"] for result in fold_results])
print(f"Average ROC AUC across folds: {avg_roc_auc:.4f}")

# Print the best ROC AUC and epoch for each fold
print("\nBest ROC AUC and Epoch for Each Fold:")
for result in fold_results:
    print(f"Fold {result['fold']}: Best ROC AUC = {result['best_roc_auc']:.4f}")
```

## Model training using all training data, no val set

This is training loop for full model training without validation set, we will use the model after 200 epochs of training as final model


```python
def save_checkpoint_full_model(epoch, model, optimizer, fold, save_dir="placeholder_folder"):
    # Create a directory for the specific fold
    fold_dir = os.path.join(save_dir, f"fold_{fold}")
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # Save the checkpoint in the fold directory
    checkpoint_path = os.path.join(fold_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
```


```python
def train_model_full(model, criterion, optimizer, train_loader, num_epochs=200, scheduler=None):
    model.to(device)

    # Initialize lists to store loss values
    train_losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        for pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age in train_loader:  
            pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age = (
                pre_ap.to(device), pre_lat.to(device), post_ap.to(device), post_lat.to(device), labels.to(device), prob_labels.to(device), male.to(device), age.to(device)
            )
            
            optimizer.zero_grad()
            outputs = model([pre_ap, pre_lat, post_ap, post_lat], male, age)  # Forward pass with 4 images
            loss = criterion(outputs, prob_labels)         # Calculate loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Adjust learning rate with scheduler (if provided)
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        print(f"  Time: {epoch_time:.2f} seconds")

        if (epoch + 1) % 25 == 0:
            save_checkpoint_full_model(epoch + 1, model, optimizer, fold = "full_model")

    return train_losses
```


```python
# Parameters
batch_size = 64
num_epochs = 200

train_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
model = MultiImageCoatNet(num_images=4, num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

train_losses = train_model_full(
    model, criterion, optimizer, train_loader, num_epochs=num_epochs, scheduler=scheduler)

print("Model training done")
```

## Check test performance

Using 5 sets of weight from 4-fold cross-validation and the final model, evaluate performance on test dataset and plot ROC curve as well as Calibration curve


```python
# Paths to model checkpoints
model = MultiImageCoatNet(num_images=4, num_classes=1).to(device)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
checkpoints = [
    'placeholder_folder/fold_1/model_epoch_200.pth',
    'placeholder_folder/fold_2/model_epoch_200.pth',
    'placeholder_folder/fold_3/model_epoch_200.pth',
    'placeholder_folder/fold_4/model_epoch_200.pth',
    'placeholder_folder/fold_full_model/model_epoch_200.pth'
]

y_true_list = []
y_pred_list = []

# Evaluate each model
for fold, checkpoint_path in enumerate(checkpoints, start=1):
    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_true, y_pred = [], []
    
    # Make predictions
    with torch.no_grad():
        for pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age in test_loader:
            pre_ap, pre_lat, post_ap, post_lat, labels, prob_labels, male, age = (
                pre_ap.to(device), pre_lat.to(device), post_ap.to(device), post_lat.to(device), labels.to(device), prob_labels.to(device), male.to(device), age.to(device)
            )

            outputs = model([pre_ap, pre_lat, post_ap, post_lat], male, age)

            y_true.extend(labels.cpu().numpy())       # Collect ground truth
            y_pred.extend(torch.sigmoid(outputs).cpu().numpy())  # Collect predictions

    # Store results for the fold
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)

    # Calculate metrics
    y_pred_binary = (y_pred > 0.5).astype(int)
    roc_auc = roc_auc_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)

    print(f"Fold {fold} Results:")
    print(f"ROC AUC = {roc_auc}")
    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1 = {f1}\n")

y_pred_avg = np.mean(y_pred_list, axis = 0)
y_true_avg = np.mean(y_true_list, axis = 0)
y_pred_avg_binary = (y_pred_avg > 0.5).astype(int)
roc_auc = roc_auc_score(y_true_avg, y_pred_avg, average="macro")
accuracy = accuracy_score(y_true_avg, y_pred_avg_binary)
precision = precision_score(y_true_avg, y_pred_avg_binary, average="macro", zero_division=0)
recall = recall_score(y_true_avg, y_pred_avg_binary, average="macro", zero_division=0)
f1 = f1_score(y_true_avg, y_pred_avg_binary, average="macro", zero_division=0)

print("Ensemble model results:")
print(f"ROC AUC = {roc_auc}")
print(f"Accuracy = {accuracy}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"F1 = {f1}\n")

# Plot the results for all folds
y_pred_list.append(y_pred_avg)
y_true_list.append(y_true_avg)
labels = [f"Fold {i+1}" for i in range(len(checkpoints))]
labels.append("Ensemble_model")
plot_multiple_roc_and_calibration_curves(y_true_list, y_pred_list, labels=labels, n_bins=10)
```
