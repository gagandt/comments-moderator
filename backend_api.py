import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
import json
import keras
from dict import check


def predict_score(test_sequence):
    max_sequence_length = 200
    model = load_model('./my_model.h5')
    with open('./word_integer_map.json') as f:
        word_to_id = json.load(f)

    temp = []
    test_sequence = test_sequence.lower()

    for word in test_sequence.split(" "):
       try:
           temp.append(word_to_id[word])
       except:
           x = 1

    temp_padded = sequence.pad_sequences([temp], maxlen=max_sequence_length)
    pred_score = model.predict(np.array([temp_padded][0]))[0][0]


    return pred_score
