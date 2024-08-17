from data_loader import load_data

def main() -> None:
    # Load the data
    train_loader, test_loader = load_data(batch_size=64)

if __name__ == '__main__':
    main()