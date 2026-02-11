# Deep Learning for Lung Cancer Prediction using Genome Sequences

This project implements deep learning models (CNN and LSTM) to predict lung cancer from genome sequences. The code has been refactored for portability and modularity.

## Project Structure

```
Deep-Learning-Technique-For-Predicting-Lung-Cancer-Using-Genome-Sequence-main/
├── src/
│   ├── data_loader.py    # Data loading, K-mer encoding, SMOTE balancing
│   ├── models.py         # CNN and LSTM model definitions
│   └── train.py          # Main training script with argument parsing
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset:**
    Ensure you have the `Final.csv` dataset containing `sequence` and `label` columns.

## Usage

Run the training script from the root directory:

```bash
python src/train.py [ARGUMENTS]
```

### Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--data_path` | Path to the dataset CSV file. | `/content/Final.csv` |
| `--model_type` | Model to train: `cnn` or `lstm`. | `cnn` |
| `--epochs` | Number of training epochs. | `10` |
| `--tune_epochs` | Number of epochs for hyperparameter tuning. | `5` |
| `--max_trials` | Max trials for Keras Tuner. | `5` |
| `--batch_size` | Batch size for training. | `64` |
| `--output_dir` | Directory to save results (plots, reports). | `results` |
| `--kmer_size` | Size of K-mers for encoding. | `3` |

### Examples

**Train a CNN model with default settings:**
```bash
python src/train.py --data_path path/to/Final.csv
```

**Train an LSTM model with custom epochs:**
```bash
python src/train.py --data_path path/to/Final.csv --model_type lstm --epochs 20
```

## features

- **Modular Design:** Separate modules for data loading, model definition, and training.
- **Portability:** Runs on any environment (local, Colab, server) by specifying the data path.
- **Hyperparameter Tuning:** Uses Keras Tuner (`RandomSearch`) to find the best model architecture.
- **Data Balancing:** Implements SMOTE to handle class imbalance.
- **Evaluation:** standardized classification report and confusion matrix plotting.
