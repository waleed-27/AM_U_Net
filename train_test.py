import torch
from metrics import calculate_metrics
from tqdm import tqdm

# Training loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': total_loss / len(dataloader)})
    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")

# Testing loop
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy, total_sensitivity, total_specificity, total_geometric_mean = 0, 0, 0, 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate metrics
            accuracy, sensitivity, specificity, geometric_mean = calculate_metrics(outputs, masks)
            total_accuracy += accuracy
            total_sensitivity += sensitivity
            total_specificity += specificity
            total_geometric_mean += geometric_mean
            
            pbar.set_postfix({'Loss': total_loss / len(dataloader)})
            
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    avg_sensitivity = total_sensitivity / len(dataloader)
    avg_specificity = total_specificity / len(dataloader)
    avg_geometric_mean = total_geometric_mean / len(dataloader)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f}, Sensitivity: {avg_sensitivity:.4f}, Specificity: {avg_specificity:.4f}, Geometric Mean: {avg_geometric_mean:.4f}")
    
    return avg_loss


