import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import cv2 


# Denoising Function 

def denoise_image(input_image):
    """
    Denoises a PIL image using OpenCV's fastNlMeansDenoisingColored.
    
    Parameters:
        input_image (PIL.Image): Input image.
        
    Returns:
        PIL.Image: Denoised image.
    """
    np_img = np.array(input_image)
    # Convert RGB to BGR (OpenCV uses BGR)
    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    # Apply denoising (adjust parameters if needed)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(np_img_bgr, None, 10, 10, 7, 21)
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(denoised_rgb)


#  Build and load your Faster R-CNN model

def get_fasterrcnn_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


#  Process a single image for head/tail detection  counts the number of Heads per image.

def process_image(model, input_image_path, output_image_path, threshold, confidence_value_threshold):
    # Load, denoise, and preprocess image
    input_image = Image.open(input_image_path).convert("RGB")
    input_image = denoise_image(input_image)
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(input_image).unsqueeze(0)  
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Extract predictions (Faster R-CNN outputs only boxes, scores, and labels)
    boxes = predictions["boxes"].cpu().numpy()    
    scores = predictions["scores"].cpu().numpy()    
    labels = predictions["labels"].cpu().numpy()    

    # Filter out low-confidence predictions for drawing
    keep_indices = scores >= threshold
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    # Convert image to numpy array for drawing
    image_np = np.array(input_image)

    # Mapping for head/tail detection:
    #  0: background, 1: Head, 2: Tail
    class_names = {
        0: "background",
        1: "Head",
        2: "Tail"
    }

    head_count = 0
    details_list = []

    # Process each detection 
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        
        # Draw bounding box (white borders)
        image_np[y1:y2, x1:x1+2, :] = 255   # left edge
        image_np[y1:y2, x2-2:x2, :] = 255     # right edge
        image_np[y1:y1+2, x1:x2, :] = 255     # top edge
        image_np[y2-2:y2, x1:x2, :] = 255     # bottom edge

        class_id = labels[i]
        class_name = class_names.get(class_id, "Unknown")
        confidence = scores[i]

        # Count as a Head if the detection is "Head" and meets the confidence threshold
        if class_name == "Head" and confidence >= confidence_value_threshold:
            head_count += 1

        details_list.append(
            f"Box{i+1}: {class_name}, score={confidence:.2f}, coords=({x1},{y1},{x2},{y2})"
        )

    # Draw text labels onto the image
    output_image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(output_image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for i in range(len(boxes)):
        class_id = labels[i]
        class_name = class_names.get(class_id, "Unknown")
        confidence = scores[i]
        x1, y1, x2, y2 = map(int, boxes[i])
        text = f"{class_name} ({confidence:.2f})"
        text_position = (x1, max(y1 - 20, 0))
        text_bbox = draw.textbbox(text_position, text, font=font)
        draw.rectangle(text_bbox, fill="white")
        draw.text(text_position, text, fill="black", font=font)

    # Save the annotated image
    output_image.save(output_image_path)
    details_str = "; ".join(details_list)
    return head_count, details_str


#  Process Dataset folder 

def process_dataset(model, input_folder, output_folder, threshold, confidence_value_threshold):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    head_counts = []  # To store head count per image
    dataset_name = os.path.basename(input_folder)
    
    # Iterate over images in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, "output_" + filename)
            if not os.path.exists(input_image_path):
                print(f"Image {input_image_path} not found, skipping.")
                continue
            head_count, details = process_image(
                model, input_image_path, output_image_path,
                threshold, confidence_value_threshold
            )
            head_counts.append(head_count)
            print(f"Processed {filename}: Head count = {head_count}")
    
    return head_counts


#  Plot distribution of head counts

def plot_head_distribution(head_counts, dataset_name="Dataset"):
    plt.figure(figsize=(8, 6))
    bins = range(min(head_counts), max(head_counts) + 2)
    plt.hist(head_counts, bins=bins, edgecolor='black', align='left')
    plt.xlabel("Head Count per Image")
    plt.ylabel("Number of Images")
    plt.title(f"Distribution of Head Counts in {dataset_name}")
    plt.xticks(bins)
    plt.tight_layout()
    plt.show()

# Main script

def main():
    num_classes = 3  # Background, Head, Tail
    model_path = "head_tail_coin_best.pth"  
    
    #  paths for Dataset 2
    input_folder = r"C:\Users\eatha\Downloads\dataset2_2025ECPb\dataset2"
    output_folder = r"C:\Users\eatha\Downloads\dataset2_2025ECPb\output\Head_Tail\dataset2"

    # Load the Faster R-CNN model
    model = get_fasterrcnn_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Process Dataset and get head counts for each image
    head_counts = process_dataset(model, input_folder, output_folder,
                                  threshold=0.5, confidence_value_threshold=0.4)
    # Plot the distribution of head counts
    plot_head_distribution(head_counts, dataset_name="Dataset2")

if __name__ == "__main__":
    main()
