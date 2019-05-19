import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
import json
from dict import check

max_sequence_length = 200 # maximum limit of comment
model = load_model('my_model.h5') # loads trained compiled model
with open('word_integer_map.json') as f:
    word_to_id = json.load(f) # loads the vocabulary dictionary

test_sequence = "This was a very irrelevent post why did you even write this, waste my whole time, would have been better if i would have masturbated instead"
test_sequence = test_sequence.lower()
save  = []
temp = []
ids = []


for word in test_sequence.split(" "):
   try:
       #if (check(word) != -1):
           #continue
       #print(word)
       temp.append(word_to_id[word])
       i = []
       i.append(word_to_id[word])
       ids.append(i)
       save.append(word)
       print(word)
   except:
       x = 1
       #print("lassan")
           
pads = []
for i in ids:
    pads.append(sequence.pad_sequences([i], maxlen=max_sequence_length))
    
scores = []
for p in pads:
    scores.append(model.predict(np.array([p][0]))[0][0])

temp_padded = sequence.pad_sequences([temp], maxlen=max_sequence_length)
pred_score = model.predict(np.array([temp_padded][0]))[0][0]
#print([temp][0][1])

temp_padded = sequence.pad_sequences([temp], maxlen=max_sequence_length)
pred_score = model.predict(np.array([temp_padded][0]))[0][0]
print("the score for the comment is %s" % (pred_score * 100))

s_dict = {}
i = -1

for s in scores:
    i+=1
    if (check(save[i]) != -1):
        continue
    s_dict[s] = save[i]
    

ans = sorted(s_dict.items(),reverse=True)

print(ans)

#print(scores)
 # gives score percentage