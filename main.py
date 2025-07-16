from src.preprocessing import load_and_prepare_data
from src.model_utils import train_and_evaluate_models

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    train_and_evaluate_models(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
