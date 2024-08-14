from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from typing import Tuple

def load_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Define the transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.ToTensor()
    ])

    # Load the training and testing datasets
    train_dataset = ImageFolder('data/train_images', transform=transform)
    test_dataset = ImageFolder('data/test_images', transform=transform)

    # Create the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader