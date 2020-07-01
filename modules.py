import numpy as np 
import tensorflow as tf 
from Settings import Config

def SelfAtt(inputs, W, a):
    W_input = tf.matmul(tf.reshape(inputs, [-1, tf.shape(inputs)[-1]]), W)
    weight_a = tf.nn.softmax(tf.tanh(tf.reshape(tf.matmul(W_input, a), [-1,1,tf.shape(inputs)[1]])))
    res =  tf.matmul(weight_a, inputs)
    res = tf.squeeze(res, 1)
    return res

def SigmoidAtt(inputs, Wr, Wm, Wu):
    #inputs: [num,len, hidden]
    H_hot = tf.multiply(tf.matmul(tf.reshape(inputs, [-1, tf.shape(inputs)[-1]]), Wu),tf.nn.sigmoid(tf.matmul(tf.reshape(inputs, [-1, tf.shape(inputs)[-1]]), Wm))) #[num*len ,atten]
    Att_a = tf.nn.softmax(tf.reshape(tf.matmul(H_hot, Wr), [tf.shape(inputs)[0], 1, tf.shape(inputs)[1]])) #[num , 1 len]
    temp_new = tf.matmul(Att_a, inputs) #[num, 1 ,hidden]
    temp_new = tf.squeeze(temp_new, 1)
    return temp_new  #[num, hidden]


def WordLevelGAT(inputs, graph, W, a, r):
    # inputs: [num, len ,hidden]
    # graph: [num,2, len, len]
    graph = tf.cast(graph, tf.float32)
    part_1 = tf.tile(tf.convert_to_tensor(inputs), [1, 1, tf.shape(inputs)[1]])
    part_1 = tf.reshape(part_1, [tf.shape(inputs)[0], -1, tf.shape(inputs)[2]]) #[num, len*len, hidden] 
    part_2 = tf.tile(tf.convert_to_tensor(inputs), [1, tf.shape(inputs)[1], 1]) #[num, len*len, hidden]

    W_part_1 = tf.reshape(tf.matmul(tf.reshape(part_1, [-1, tf.shape(inputs)[2]]), W), [tf.shape(part_1)[0], tf.shape(part_1)[1], -1]) #[num,len*len, atten]
    W_part_2 = tf.reshape(tf.matmul(tf.reshape(part_2, [-1, tf.shape(inputs)[2]]), W), [tf.shape(part_2)[0], tf.shape(part_2)[1], -1]) #[num,len*len, atten]

    graph1 = tf.reshape(graph[:,0,:,:], [tf.shape(graph)[0],-1, 1])  #[num, len*len,1]
    concat_vec = tf.concat([W_part_1, W_part_2], -1)   #[num, len*len, 2atten]
    concat_vec = tf.multiply(concat_vec, graph1)

    tmp_w = tf.nn.tanh(tf.matmul(tf.reshape(concat_vec, [-1, tf.shape(concat_vec)[-1]]), a )) #[num*len*len, 1]
    tmp_w = tf.reshape(tmp_w, [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[1]]) #[num, len, len]
    graph2 = tf.reshape(graph[:,1,:,:], [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[1]])
    tmp_w = tf.add(tmp_w, graph2)  #[num, len, len]

    weight_g = tf.nn.softmax(tf.reshape(tmp_w, [tf.shape(inputs)[0], tf.shape(inputs)[1], 1, tf.shape(inputs)[1]])) #[num, len,1, len]
    weight_g = tf.add((1-r)*weight_g, r*tf.reshape(graph[:,0,:,:], [tf.shape(inputs)[0], tf.shape(inputs)[1], 1, tf.shape(inputs)[1]]))
    temp_enc = tf.matmul(weight_g, tf.reshape(W_part_2, [tf.shape(inputs)[0],tf.shape(inputs)[1],tf.shape(inputs)[1],-1])) #[num, len,1, atten]
    temp_enc = tf.squeeze(temp_enc, 2) #[num, len, atten]
    return temp_enc




def SenLevelGAT(all_nodes, W, a, r):
    #a strongly connected graph
    temp_length = tf.shape(all_nodes)[0]

    part_1 = tf.tile(all_nodes, [1, temp_length])
    part_1 = tf.reshape(part_1, [-1, tf.shape(all_nodes)[-1]]) #[len*len, 2hidden]
    part_2 = tf.tile(tf.convert_to_tensor(all_nodes), [temp_length, 1]) #[len*len, 2hidden]

    W_all_nodes = tf.matmul(all_nodes, W) #[len, atten]
    W_part_1 = tf.matmul(part_1, W) #[len*len, atten]
    W_part_2 = tf.matmul(part_2, W) #[len*len, atten]

    concat_vec = tf.concat([W_part_1, W_part_2], 1) #[len, 2atten]
    weight_g = tf.nn.softmax(tf.reshape(tf.nn.tanh(tf.matmul(concat_vec, a)), [temp_length, temp_length])) #[len*len]
    weight_g = tf.add((1-r)*weight_g, r*tf.ones([temp_length, temp_length]))
    temp_enc = tf.matmul(weight_g, W_all_nodes) #[len, atten]
    return temp_enc
