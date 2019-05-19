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
    save  = []
    temp = []
    ids = []


    for word in test_sequence.split(" "):
       try:
           if (check(word) != -1):
               continue
           i = []
           i.append(word_to_id[word])
           ids.append(i)
           save.append(word)
           #print(word)
       except:
           x = 1

    pads = []
    for i in ids:
        pads.append(sequence.pad_sequences([i], maxlen=max_sequence_length))

    scores = []
    for p in pads:
        scores.append(model.predict(np.array([p][0]))[0][0])

    s_dict = {}
    i = -1
    for s in scores:
        i+=1
        print(i)
        if (check(save[i]) != -1):
            continue
        s_dict[s] = save[i]


    ans = sorted(s_dict.items(),reverse=True)

    return  ans
