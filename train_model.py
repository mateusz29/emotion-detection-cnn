from data_loader import load_data
import torch
from CNN import CNN

def main() -> None:
    # Load the training data
    train_loader, _ = load_data(batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)

    # Defining the hyper parameters
    learning_rate = 0.001
    weight_decay = 0.01
    num_epochs = 25
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move tensors to the configured device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'model/emotion_cnn.pth')

if __name__ == '__main__':
    main()