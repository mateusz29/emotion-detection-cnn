from data_loader import load_data
import torch

def main() -> None:
    # Load the data
    train_loader, test_loader = load_data(batch_size=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    main()