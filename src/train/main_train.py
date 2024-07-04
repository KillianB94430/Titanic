
from load_and_preprocess_data import load_and_preprocess_data
from train_and_evaluate_model import train_and_evaluate_model


def main():
    print("Loading and preprocessing data...")
    load_and_preprocess_data()
    print("Training and evaluating model...")
    train_and_evaluate_model()

if __name__ == "__main__":
    main()
