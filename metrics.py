# Function to calculate metrics
def calculate_metrics(outputs, masks):
    # Binarize outputs and masks
    outputs = (outputs > 0.5).float()
    masks = (masks > 0.5).float()

    TP = (outputs * masks).sum().item()
    TN = ((1 - outputs) * (1 - masks)).sum().item()
    FP = (outputs * (1 - masks)).sum().item()
    FN = ((1 - outputs) * masks).sum().item()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Calculate geometric mean
    geometric_mean = (sensitivity * specificity) ** 0.5 if (sensitivity + specificity) > 0 else 0

    return accuracy, sensitivity, specificity, geometric_mean
