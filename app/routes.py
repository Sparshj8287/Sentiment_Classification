from flask import render_template,redirect,url_for,flash,request
from wtforms.validators import ValidationError
from app import app
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, GRU, Embedding
from keras.layers import Activation, Bidirectional, GlobalMaxPool1D, GlobalMaxPool2D, Dropout
from keras.models import Model
from keras.preprocessing import text, sequence
import transformers
from transformers import AutoTokenizer
from tokenizers import BertWordPieceTokenizer
from keras.initializers import Constant
import numpy as np
import re
import tensorflow as tf
import os
os.environ['hf_hub_download'] = './.cache'
@app.route('/')
def home_page():
    return render_template('index.html')


tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowercase=True)



def fast_encode_sentence(text, tokenizer, maxlen=128):    
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length=maxlen)
    all_ids = []
    
    text_chunk = text
    encs = tokenizer.encode(text_chunk)
    all_ids.extend([encs.ids])
    
    return np.array(all_ids)



  
  
transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')

embedding_size = 128
inp = Input(shape=(128, ))
embedding_matrix=transformer_layer.weights[0].numpy()
x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],embeddings_initializer=Constant(embedding_matrix),trainable=False)(inp)
x = Bidirectional(LSTM(25, return_sequences=True,recurrent_regularizer='L1L2'))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.9)(x)
x = Dense(50, activation='relu',kernel_initializer='he_normal',kernel_regularizer="L1L2")(x)
x = Dropout(0.9)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inp], outputs=x)
model.load_weights('distilbert_model_weights.best.hdf5')


def predict_on_sentence(model,text):
  text=text.lower()
  pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  text = pattern.sub('', text)
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"he's", "he is", text)
  text = re.sub(r"she's", "she is", text)
  text = re.sub(r"that's", "that is", text)        
  text = re.sub(r"what's", "what is", text)
  text = re.sub(r"where's", "where is", text) 
  text = re.sub(r"\'ll", " will", text)  
  text = re.sub(r"\'ve", " have", text)  
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"don't", "do not", text)
  text = re.sub(r"did't", "did not", text)
  text = re.sub(r"can't", "can not", text)
  text = re.sub(r"it's", "it is", text)
  text = re.sub(r"couldn't", "could not", text)
  text = re.sub(r"have't", "have not", text)
  text=re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
  text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
  text = re.sub(r'(.)\1{3,}',r'\1', text)
  final_text=fast_encode_sentence(text,fast_tokenizer)
  prediction=model.predict(final_text)
  final_text=tf.squeeze(tf.round(prediction))
  return final_text


@app.route('/predict',methods=['POST'])
def predict():
    int_features = request.form.get("sentence")
    int_features=str(int_features)
    final_result=predict_on_sentence(model,int_features)
    result='bad'
    if(final_result==1):
        result='good'
    return render_template('index.html', prediction_text='This is a {} comment'.format(result))

