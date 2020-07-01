import numpy as np
from bert_serving.client import BertClient
bc = BertClient()
beauty_sentence = []
with open('./data/beauty/beauty-que-sub.output', 'r+') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.split('\n')[0]
        beauty_sentence.append(line)
    #print (beauty_sentence)
beauty_embedding = bc.encode(beauty_sentence)
print (np.shape(beauty_embedding))
fin = []
for item in beauty_embedding:
    fin.append(item[1:-1][:])
print (np.shape(fin))
np.save('./data/beauty/beauty-que-15.npy', fin)
