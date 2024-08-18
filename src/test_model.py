from data_loader import load_data
import torch
from CNN import CNN

def main() -> None:
    # Load the testing data
    _, test_loader = load_data(batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.load_state_dict(torch.load('./model/emotion_cnn.pth', weights_only=True))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # Get the class with the highest probability
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()