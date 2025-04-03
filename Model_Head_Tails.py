import os
import torch
import torch.utils.data
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pycocotools.coco as coco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import cv2  


# Custom Dataset Class for Faster R-CNN in COCO format
class CoinCocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        """
        Initializes the dataset using COCO-format annotations.
        :param img_folder: Folder where images are stored.
        :param ann_file: COCO-format JSON annotations file.
        :param transforms: Transformations to apply to each image.
        """
        self.img_folder = img_folder
        self.coco = coco.COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        # Return the total number of images.
        return len(self.ids)

    def __getitem__(self, index):
        # Get the image ID and load annotations for that image.
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(img_id)
        annotations = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Parse bounding boxes and labels.
        boxes, labels = [], []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        # Add extra keys required by Faster R-CNN.
        target["image_id"] = torch.tensor([img_id])
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["area"] = area
            target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            target["area"] = torch.tensor([])
            target["iscrowd"] = torch.tensor([])

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


# Data Transforms

def get_transforms(train=True):
    """
    Returns a composed transform. For training, additional augmentations
    (like random Gaussian blur) are applied.
    """
    t_list = [T.ToTensor()]
    if train:
        t_list.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1))], p=0.25))
    return T.Compose(t_list)


# Build the Faster R-CNN Model

def get_fasterrcnn_model(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and replaces its classification head
    to match the number of classes.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Evaluate mAP and Average Recall (AR)

def evaluate_map(model, data_loader, device):
    """
    Evaluates the Mean Average Precision (mAP) and Average Recall (AR) on a dataset.
    Returns the mAP and AR (using 'mar_100' from the computed metric).
    """
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            metric.update(preds, targets)
    result = metric.compute()
    map_val = result['map'].item()
    ar_val = result['mar_100'].item() if 'mar_100' in result else None
    return map_val, ar_val


# Main Training and Evaluation Script

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    num_classes = 3  #  Background (0), Head (1), Tail (2)

    # Dataset paths
    train_dir = "Heads_Tails_Dataset/train"
    val_dir = "Heads_Tails_Dataset/valid"
    test_dir = "Heads_Tails_Dataset/test"
    train_ann_file = os.path.join(train_dir, "_annotations.coco.json")
    val_ann_file = os.path.join(val_dir, "_annotations.coco.json")
    test_ann_file = os.path.join(test_dir, "_annotations.coco.json")

    # Create datasets with transforms
    train_dataset = CoinCocoDataset(train_dir, train_ann_file, transforms=get_transforms(train=True))
    val_dataset = CoinCocoDataset(val_dir, val_ann_file, transforms=get_transforms(train=False))
    test_dataset = CoinCocoDataset(test_dir, test_ann_file, transforms=get_transforms(train=False))
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # Debug: Check label distribution for a few samples
    label_counts = {0: 0, 1: 0, 2: 0}
    for i in range(min(10, len(train_dataset))):
        _, target = train_dataset[i]
        for label in target['labels']:
            label_counts[label.item()] += 1
    print(f"Label distribution in first 10 samples: {label_counts}")

    # Data loaders with collate function
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Build and move model to device
    model = get_fasterrcnn_model(num_classes)
    model.to(device)

    # Set up optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 20
    best_val_map = 0.0
    # Lists to record metrics for plotting
    train_loss_list = []
    train_map_list = []
    train_ar_list = []
    val_map_list = []
    val_ar_list = []
    test_map_list = []
    test_ar_list = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        train_loss = running_loss / len(train_loader)
        train_loss_list.append(train_loss)

        # Evaluate mAP and AR on training, validation, and test sets
        train_map, train_ar = evaluate_map(model, train_loader, device)
        val_map, val_ar = evaluate_map(model, val_loader, device)
        test_map, test_ar = evaluate_map(model, test_loader, device)
        train_map_list.append(train_map)
        train_ar_list.append(train_ar)
        val_map_list.append(val_map)
        val_ar_list.append(val_ar)
        test_map_list.append(test_map)
        test_ar_list.append(test_ar)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, "
              f"Train mAP: {train_map:.4f}, Train AR: {train_ar:.4f}, "
              f"Val mAP: {val_map:.4f}, Val AR: {val_ar:.4f}, "
              f"Test mAP: {test_map:.4f}, Test AR: {test_ar:.4f}")
        scheduler.step()

        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), "head_tail_coin_best.pth")
            print(f"Saved best model with Val mAP: {val_map:.4f}")

    print("Training complete.")
    torch.save(model.state_dict(), "head_tail_coin_final.pth")

    # Plot mAP curves
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_map_list, label="Train mAP")
    plt.plot(epochs, val_map_list, label="Val mAP")
    plt.plot(epochs, test_map_list, label="Test mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Average Recall (AR) curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ar_list, label="Train AR")
    plt.plot(epochs, val_ar_list, label="Val AR")
    plt.plot(epochs, test_ar_list, label="Test AR")
    plt.xlabel("Epoch")
    plt.ylabel("Average Recall")
    plt.title("Average Recall over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
