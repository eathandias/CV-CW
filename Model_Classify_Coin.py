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

# Custom Dataset Class (COCO format) for Faster R-CNN
class CoinCocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.coco = coco.COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Load image and annotations
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(img_id)
        annotations = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Parse bounding boxes and labels
        boxes, labels = [], []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        
        # Extra keys required by Faster R-CNN
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
    t_list = [T.ToTensor()]
    if train:
        # Applying a random Gaussian blur 20% of the time
        t_list.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1))], p=0.2))
    return T.Compose(t_list)

# Build the Faster R-CNN Model
def get_fasterrcnn_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Evaluate mAP on a dataset
def evaluate_map(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            metric.update(preds, targets)
    result = metric.compute()
    return result['map'].item()

# Main training script 
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    num_classes = 8  # For coin detection: 1 background + 7 coin classes

    # Dataset paths 
    train_dir = "Uk_Coin/train"
    val_dir = "Uk_Coin/valid"
    test_dir = "Uk_Coin/test"
    train_ann_file = os.path.join(train_dir, "_annotations.coco.json")
    val_ann_file = os.path.join(val_dir, "_annotations.coco.json")
    test_ann_file = os.path.join(test_dir, "_annotations.coco.json")

    # Create datasets
    train_dataset = CoinCocoDataset(train_dir, train_ann_file, transforms=get_transforms(train=True))
    val_dataset = CoinCocoDataset(val_dir, val_ann_file, transforms=get_transforms(train=False))
    test_dataset = CoinCocoDataset(test_dir, test_ann_file, transforms=get_transforms(train=False))
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # Check label distribution in first few samples (for debugging)
    label_counts = {i: 0 for i in range(num_classes)}
    for i in range(min(10, len(train_dataset))):
        _, target = train_dataset[i]
        for label in target['labels']:
            label_counts[label.item()] += 1
    print(f"Label distribution in first 10 samples: {label_counts}")

    # Create data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Build the model and move to device
    model = get_fasterrcnn_model(num_classes)
    model.to(device)

    # Set up optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 50
    best_val_map = 0.0
    early_stop_counter = 0
    patience = 5  # Early stopping patience

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
        
        # Evaluate mAP on train, validation, and test sets
        train_map = evaluate_map(model, train_loader, device)
        val_map = evaluate_map(model, val_loader, device)
        test_map = evaluate_map(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, "
              f"Train mAP: {train_map:.4f}, Val mAP: {val_map:.4f}, Test mAP: {test_map:.4f}")
        scheduler.step()

        # Early stopping logic
        if val_map > best_val_map:
            best_val_map = val_map
            early_stop_counter = 0  # reset counter if improvement occurs
            torch.save(model.state_dict(), "Coin_Classify_Best.pth")
            print(f"Saved best model with Val mAP: {val_map:.4f}")
        else:
            early_stop_counter += 1
            print(f"No improvement in Val mAP for {early_stop_counter} epoch(s).")
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    torch.save(model.state_dict(), "Coin_Classify_final.pth")

    # Print final evaluation metrics
    final_train_map = evaluate_map(model, train_loader, device)
    final_val_map = evaluate_map(model, val_loader, device)
    final_test_map = evaluate_map(model, test_loader, device)
    print(f"Final Train mAP: {final_train_map:.4f}")
    print(f"Final Val mAP: {final_val_map:.4f}")
    print(f"Final Test mAP: {final_test_map:.4f}")

if __name__ == "__main__":
    main()
