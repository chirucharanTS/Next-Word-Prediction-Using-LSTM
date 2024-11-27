import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len, temperature=1.0):
    # Convert input text to a sequence
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    # Predict probabilities
    predicted = model.predict(token_list, verbose=0)[0]
    
    # Apply temperature scaling
    scaled_probs = np.exp(np.log(predicted) / temperature)
    scaled_probs = scaled_probs / np.sum(scaled_probs)
    
    # Sample a word index from the probability distribution
    predicted_word_index = np.random.choice(range(len(scaled_probs)), p=scaled_probs)
    
    # Map index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "No prediction"


# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

