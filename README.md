# Next-Word-Prediction-Using-LSTM
This project aims to build a deep learning model that predicts the next word in a sequence using Long Short-Term Memory (LSTM) networks. The model is trained on Shakespeare's Hamlet text and is capable of generating predictions based on a given sequence of words.

### README for GitHub

---
## Project Overview

This project involves the following steps:
1. **Data Collection**: The text of Shakespeare's *Hamlet* is used as the dataset.
2. **Data Preprocessing**: Tokenizing the text into words, creating input sequences, and padding them for uniformity.
3. **Model Building**: An LSTM model with an embedding layer, two LSTM layers, and a softmax output layer.
4. **Model Training**: Training the model with early stopping and learning rate scheduling.
5. **Model Evaluation**: Evaluating the model on example sentences.
6. **Deployment**: A Streamlit web app that allows real-time predictions of the next word.

## Directory Structure

```
/Next-Word-Prediction-LSTM
    /data
        hamlet.txt                - Text file containing the Hamlet dataset.
    /model
        next_word_lstm.h5          - Saved LSTM model.
        tokenizer.pickle           - Saved tokenizer object.
    /app
        streamlit_app.py           - Streamlit app for real-time predictions.
    /scripts
        train_model.py             - Script to train the LSTM model.
    README.md                      - Project description and setup instructions.
```

## Project Workflow

1. **Data Collection**: The text of *Hamlet* is loaded from the Gutenberg corpus and saved locally as `hamlet.txt`.
2. **Preprocessing**: The text is tokenized, sequences are created, and padding is applied to ensure uniform input length.
3. **Model Building**: The model is built using two LSTM layers, an embedding layer, dropout for regularization, and a softmax layer to predict the next word.
4. **Training**: The model is trained using early stopping and learning rate scheduling to improve convergence and prevent overfitting.
5. **Prediction**: A function is implemented to predict the next word in a given sequence.
6. **Deployment**: A simple Streamlit app is provided to interact with the model in real-time.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Next-Word-Prediction-LSTM.git
   cd Next-Word-Prediction-LSTM
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Features

- **Real-time Next Word Prediction**: Enter a sequence of words, and the model predicts the next word.
- **Model Visualization**: Visualize training and validation loss and accuracy graphs.
- **Temperature-based Sampling**: Allows tuning the randomness of predictions via temperature scaling.

## Dependencies

- **TensorFlow** (for LSTM model building and training)
- **Keras** (for deep learning components)
- **Streamlit** (for the web app interface)
- **nltk** (for text processing and tokenization)
- **Matplotlib** (for plotting graphs)

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
tensorflow==2.11.0
keras==2.11.0
streamlit==1.16.0
nltk==3.7
matplotlib==3.5.1
```

## Usage

1. **Train the model**:
   Run `train_model.py` to train the model on the `hamlet.txt` dataset.
   
   ```bash
   python scripts/train_model.py
   ```

2. **Predictions**:
   After training, use the `predict_next_word` function to predict the next word in a sequence.

   Example usage in Python:
   ```python
   input_text = "To be or not to be"
   next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len, temperature=1.0)
   print(f"Next Word Prediction: {next_word}")
   ```

3. **Run the Streamlit app**:
   Start the Streamlit app for real-time word predictions.

   ```bash
   streamlit run app/streamlit_app.py
   ```

## Limitations

- The model only predicts the next word in a sequence and is limited by the quality of training data.
- Shakespeare's language is unique, which may limit the model's ability to generalize well to other types of text.
- **Data Limitations**: Only one text (Hamlet) is used for training, limiting the variety of the language model.

## Future Enhancements

1. **Expand Dataset**: Train the model on a larger corpus or multiple works of Shakespeare to improve generalization.
2. **Fine-tune Hyperparameters**: Experiment with hyperparameter tuning for better accuracy and model performance.
3. **Real-time Sentence Completion**: Extend the model to predict an entire sentence rather than just the next word.
4. **Advanced Models**: Implement transformer-based models such as GPT or BERT for improved performance.

---
