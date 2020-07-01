import numpy as np 
import re 

def cal_len(path):
    res = []
    with open(path, 'r+') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = line[:-1].split()
            res.append(len(temp))
    return res


data = np.load('./beauty/beauty-que-parse.npy')
cur_len = cal_len('./beauty/beauty-que-sub.output')
graph_length = 15  # sub-question: 15   sub-answer: 20
res = []
for i in range(len(data)):
    temp = []
    for item in data[i]:
        tt = re.findall(r"-.*?\d+",item)
        temp.append([int(tt[0][1:]), int(tt[1][1:])])
    res.append(temp)

vec = []
for i in range(len(res)):
    temp_vec = list(np.zeros((cur_len[i], cur_len[i]), dtype=np.int32))
    for item in res[i]:
        try:
            if item[0] != 0:
                temp_vec[item[0]-1][item[1]-1] = 1 
                #temp_vec[item[1]-1][item[0]-1] = 1   
        except:
             print (i)    
    vec.append(temp_vec)

for i in range(len(vec)):
    for j in range(len(vec[i])):
        vec[i][j][j] = 1


graph = list(np.zeros((len(vec),graph_length, graph_length), dtype= np.int32))
for i in range(len(vec)):
    for ii in range(min(len(vec[i]),graph_length)):
        for jj in range(min(len(vec[i]),graph_length)):
            graph[i][ii][jj] += vec[i][ii][jj]


graph1 = list(np.zeros((len(vec),graph_length, graph_length), dtype= np.float32))
for i in range(len(graph1)):
    for ii in range(graph_length):
        for jj in range(graph_length):
            if graph[i][ii][jj] == 0:
                graph1[i][ii][jj] = float("-inf")
            else:
                graph1[i][ii][jj] = 0.0
print (graph[0][0])
print (graph1[0][0])

allgraph = []
for i in range(len(graph1)):
    tmp = []
    tmp =[graph[i], graph1[i]]
    allgraph.append(tmp)
print (np.shape(allgraph))
np.save('./beauty/beauty-que-allgraph.npy', allgraph)
