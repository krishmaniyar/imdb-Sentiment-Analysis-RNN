# IMDB Sentiment Analysis using Simple RNN

A deep learning project that performs sentiment analysis on IMDB movie reviews using a Simple Recurrent Neural Network (RNN) with TensorFlow/Keras. The project includes a Streamlit web application for real-time sentiment prediction.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [License](#license)

## 🎯 Overview

This project implements a sentiment analysis model that classifies IMDB movie reviews as either positive or negative. The model is built using a Simple RNN architecture with an embedding layer and achieves good performance on the IMDB dataset.

## ✨ Features

- **Simple RNN Architecture**: Uses SimpleRNN layers with ReLU activation for sentiment classification
- **Word Embeddings**: Implements word embedding layers to convert text into dense vector representations
- **Interactive Web App**: Streamlit-based interface for real-time sentiment prediction
- **Pre-trained Model**: Includes a trained model (`simple_rnn_imdb.h5`) ready for inference
- **Comprehensive Notebooks**: Jupyter notebooks demonstrating embedding concepts, model training, and prediction

## 📁 Project Structure

```
imdb-Sentiment-Analysis-RNN/
├── main.py                    # Streamlit web application
├── simple_rnn_imdb.h5         # Pre-trained RNN model
├── simplernn.ipynb            # Model training notebook
├── embedding.ipynb            # Word embedding demonstration
├── prediction.ipynb           # Model prediction examples
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── LICENSE                    # License information
└── .gitignore                # Git ignore file
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/imdb-Sentiment-Analysis-RNN.git
cd imdb-Sentiment-Analysis-RNN
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Streamlit App

Launch the web application for interactive sentiment analysis:

```bash
streamlit run main.py
```

The app will open in your browser where you can:
1. Enter a movie review in the text area
2. Click the "Classify" button
3. View the sentiment (Positive/Negative) and prediction score

### Using the Jupyter Notebooks

**1. Training the Model (`simplernn.ipynb`)**
- Load and preprocess the IMDB dataset
- Build and train the Simple RNN model
- Save the trained model

**2. Understanding Embeddings (`embedding.ipynb`)**
- Learn about word embeddings and one-hot encoding
- Visualize embedding representations

**3. Making Predictions (`prediction.ipynb`)**
- Load the pre-trained model
- Make predictions on custom text inputs
- Decode reviews from integer sequences

## 🏗️ Model Architecture

The model consists of the following layers:

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding_1 (Embedding)     (None, 500, 128)          1,280,000   
simple_rnn_1 (SimpleRNN)    (None, 128)               32,896     
dense_1 (Dense)             (None, 1)                 129       
=================================================================
Total params: 1,313,025 (5.01 MB)
Trainable params: 1,313,025 (5.01 MB)
Non-trainable params: 0 (0.00 Byte)
```

### Key Components:

- **Embedding Layer**: Converts word indices to 128-dimensional dense vectors (vocabulary size: 10,000)
- **SimpleRNN Layer**: Processes sequences with 128 units and ReLU activation
- **Dense Output Layer**: Single neuron with sigmoid activation for binary classification
- **Input Length**: Fixed sequence length of 500 words

### Training Configuration:

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 10 (with early stopping)
- **Validation Split**: 20%

## 📊 Results

The model achieves competitive performance on the IMDB sentiment analysis task:

- Training accuracy progressively improves across epochs
- Validation accuracy reaches approximately 80%
- Early stopping prevents overfitting by monitoring validation loss
- The model successfully classifies movie reviews with reasonable accuracy

## 🛠️ Technologies Used

- **Python 3.11+**
- **TensorFlow 2.15.0**: Deep learning framework
- **Keras**: High-level neural networks API
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **Matplotlib**: Data visualization

## 📝 How It Works

1. **Data Preprocessing**: 
   - Reviews are converted to sequences of integers (word indices)
   - Sequences are padded to a fixed length of 500
   
2. **Training**:
   - The model learns word embeddings and sequence patterns
   - RNN processes sequences to capture temporal dependencies
   
3. **Prediction**:
   - User input is tokenized and preprocessed
   - The model outputs a probability score (0-1)
   - Score > 0.5 indicates positive sentiment, otherwise negative

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📧 Contact

For questions or feedback, please open an issue on the GitHub repository.

---

**Note**: This is an educational project demonstrating sentiment analysis using Simple RNN. For production applications, consider using more advanced architectures like LSTM, GRU, or Transformer-based models.
