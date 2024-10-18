import argparse
import os
import torch
from models import AM_U_Net, R2_AM_U_Net
from dataset import RetinalDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Vessel Segmentation Inference")
    parser.add_argument("--model", choices=["AM_U_Net", "R2_AM_U_Net"], required=True, help="Model architecture to use")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output segmented image")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the input images")
    return parser.parse_args()

def load_model(model_name, checkpoint_path, device):
    # Load the model architecture
    if model_name == "AM_U_Net":
        model = AM_U_Net()
    elif model_name == "R2_AM_U_Net":
        model = R2_AM_U_Net()
    
    # Load the checkpoint
    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path, image_size):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Convert the image to a numpy array for preprocessing
    image_np = np.array(image)

    # Extract the green channel and apply CLAHE as in RetinalDataset's preprocessing
    green_channel = image_np[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green_channel)
    rgb_image = np.stack((enhanced_green, enhanced_green, enhanced_green), axis=-1)

    # Convert the processed image back to PIL
    processed_image = Image.fromarray(rgb_image)

    # Apply transformation to resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return transform(processed_image).unsqueeze(0)  # Add batch dimension

def postprocess_output(output, image_size, save_path):
    # Squeeze to remove unnecessary dimensions
    output = output.squeeze().cpu().detach().numpy()

    # Ensure the values are between 0 and 1
    output = np.clip(output, 0, 1)

    # Convert the model output to an image (0-255 range for saving)
    output_image = (output * 255).astype(np.uint8)

    # Convert the NumPy array to a PIL Image
    output_image = Image.fromarray(output_image)

    # Resize to original image size if needed
    output_image = output_image.resize((image_size, image_size))

    # Save the output image
    output_image.save(save_path)
    print(f"Segmented image saved to {save_path}")

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model from checkpoint
    model = load_model(args.model, args.checkpoint, device)

    # Preprocess the input image
    image_tensor = preprocess_image(args.image_path, args.image_size).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        output = output.cpu()

    # Inspect the range of values in the output

    
    # Postprocess and save the output
    postprocess_output(output, args.image_size, args.output_path)

if __name__ == "__main__":
    main()
