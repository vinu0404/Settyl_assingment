

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and the tokenizers
model = tf.keras.models.load_model('status_prediction_model_deep_rnn.pkl')
with open('tokenizer_deeprnn.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder_deeprnn.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)

# Define a request body model
class Status(BaseModel):
    external_status: str

# Initialize the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post('/predict')
def predict(status: Status):
    # Preprocess the input
    sequences = tokenizer.texts_to_sequences([status.external_status])
    padded_sequences = pad_sequences(sequences, padding='post')

    # Make a prediction
    prediction = model.predict(padded_sequences)

    # Convert the prediction to the corresponding internal status label
    internal_status = label_tokenizer.index_word[prediction.argmax() + 1]

    # Return the prediction
    return {'internal_status': internal_status}

