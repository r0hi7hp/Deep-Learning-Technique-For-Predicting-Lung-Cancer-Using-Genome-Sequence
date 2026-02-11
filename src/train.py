import argparse
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Import our modules
from data_loader import load_and_preprocess_data
from models import build_cnn_model, build_lstm_model

class TestPerformance(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_acc = []
        self.test_loss = []

    def on_epoch_end(self, epoch, logs=None):
        X_test, y_test = self.test_data
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        print(f" - test_loss: {loss:.4f} - test_acc: {acc:.4f}")

def plot_history(history, test_callback, output_dir):
    epochs_range = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(epochs_range, test_callback.test_acc, label='Test Accuracy', linestyle='--')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history.history['loss'], label='Train Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.plot(epochs_range, test_callback.test_loss, label='Test Loss', linestyle='--')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Cancerous', 'Cancerous'], 
                yticklabels=['Non-Cancerous', 'Cancerous'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Deep Learning Models for Lung Cancer Prediction')
    parser.add_argument('--data_path', type=str, default='/content/Final.csv', help='Path to the dataset CSV file')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'lstm'], default='cnn', help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (after tuning)')
    parser.add_argument('--tune_epochs', type=int, default=5, help='Number of epochs for hyperparameter tuning')
    parser.add_argument('--max_trials', type=int, default=5, help='Max trials for tuner')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--kmer_size', type=int, default=3, help='Size of K-mers')

    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Starting training with {args.model_type.upper()} model...")
    
    # Load Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), maxlen, vocab_size = load_and_preprocess_data(args.data_path, args.kmer_size)

    # Callbacks
    test_callback = TestPerformance((X_test, y_test))
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # Model Builder Selection
    if args.model_type == 'cnn':
        model_builder = lambda hp: build_cnn_model(hp, maxlen, vocab_size)
        project_name = 'cnn_kmer_gene'
    else:
        model_builder = lambda hp: build_lstm_model(hp, maxlen, vocab_size)
        project_name = 'lstm_kmer_gene'

    # Tuning
    print("Starting Hyperparameter Tuning...")
    tuner = RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=args.max_trials,
        executions_per_trial=1,
        directory=os.path.join(args.output_dir, 'tuner_dir'),
        project_name=project_name
    )

    tuner.search(X_train, y_train,
                 epochs=args.tune_epochs,
                 batch_size=args.batch_size,
                 validation_data=(X_val, y_val),
                 callbacks=[early_stop])

    # Get Best Model
    best_model = tuner.get_best_models(num_models=1)[0]
    print("Best hyperparameters found.")
    tuner.results_summary(1)

    # Train Best Model
    print("Training Best Model...")
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, test_callback]
    )

    # Evaluation
    print(f"Final Test Accuracy: {test_callback.test_acc[-1]:.4f}")
    
    y_pred_probs = best_model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    report = classification_report(y_test, y_pred, digits=4)
    print("Classification Report:\n", report)
    
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Plotting
    plot_history(history, test_callback, args.output_dir)
    plot_confusion_matrix(y_test, y_pred, args.output_dir)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
