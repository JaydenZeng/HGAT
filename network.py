import tensorflow as tf 
import numpy as np 
from Settings import Config
from modules import  WordLevelGAT,SenLevelGAT, SigmoidAtt
import sys

class GATQA:
    def __init__(self, is_training):
        self.config = Config()
        self.input_question = tf.placeholder(dtype = tf.float32, shape=[None, self.config.max_sub_question_len,  768], name='input_question')
        self.input_answer = tf.placeholder(dtype=tf.float32, shape = [None, self.config.max_sub_answer_len,  768], name='input_answer')
        self.input_label = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name = 'input_label')
        self.cur_question_len = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name='cur_quesiton')
        self.cur_answer_len = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name = 'cur_answer')
        self.input_question_graph = tf.placeholder(dtype = tf.int32, shape=[None,2, self.config.max_sub_question_len, self.config.max_sub_question_len], name = 'input_question_graph')
        self.input_answer_graph = tf.placeholder(dtype = tf.int32, shape = [None,2, self.config.max_sub_answer_len, self.config.max_sub_answer_len], name = 'input_answer_graph')
       
        Hq = self.input_question
        Ha = self.input_answer
   

        lstm_fw_cell_q = tf.nn.rnn_cell.LSTMCell(self.config.lstm_hidden_size)
        lstm_bw_cell_q = tf.nn.rnn_cell.LSTMCell(self.config.lstm_hidden_size)
        lstm_fw_cell_a = tf.nn.rnn_cell.LSTMCell(self.config.lstm_hidden_size)
        lstm_bw_cell_a = tf.nn.rnn_cell.LSTMCell(self.config.lstm_hidden_size)

        if is_training:
            Hq = tf.layers.dropout(Hq, rate= self.config.keep_prob)
            Ha = tf.layers.dropout(Ha, rate= self.config.keep_prob)
            lstm_fw_cell_q = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_q, output_keep_prob = self.config.keep_prob)
            lstm_bw_cell_q = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_q, output_keep_prob = self.config.keep_prob)
            lstm_fw_cell_a = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_a, output_keep_prob = self.config.keep_prob)
            lstm_bw_cell_a = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_a, output_keep_prob = self.config.keep_prob)

        #question encoding
        with tf.variable_scope('enc_q', reuse = tf.AUTO_REUSE):
            enc_q_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell_q,
                                                            cell_bw = lstm_bw_cell_q,
                                                            inputs = Hq,
                                                            dtype = tf.float32,
                                                            time_major = False)
            Hq = tf.concat([enc_q_outputs[0], enc_q_outputs[1]], axis=2)

        #answer encoding
        with tf.variable_scope('enc_a', reuse = tf.AUTO_REUSE):
            enc_a_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell_a,
                                                            cell_bw = lstm_bw_cell_a,
                                                            inputs = Ha,
                                                            dtype = tf.float32,
                                                            time_major = False)
            Ha = tf.concat([enc_a_outputs[0], enc_a_outputs[1]], axis=2)
     
        
        with tf.variable_scope('all_weights', reuse = tf.AUTO_REUSE):
            W_sq1 = tf.get_variable('W_sq1', [self.config.head_num, self.config.hidden_size, self.config.hidden_size/self.config.head_num])
            a_sq1 = tf.get_variable('a_sq1', [self.config.head_num, 2*self.config.hidden_size/self.config.head_num, 1])
            W_sa1 = tf.get_variable('W_sa1', [self.config.head_num, self.config.hidden_size, self.config.hidden_size/self.config.head_num])
            a_sa1 = tf.get_variable('a_sa1', [self.config.head_num, 2*self.config.hidden_size/self.config.head_num, 1])
       
            W_sq2 = tf.get_variable('W_sq2', [self.config.head_num, self.config.hidden_size, self.config.hidden_size/self.config.head_num])
            a_sq2 = tf.get_variable('a_sq2', [self.config.head_num, 2*self.config.hidden_size/self.config.head_num, 1])
            W_sa2 = tf.get_variable('W_sa2', [self.config.head_num, self.config.hidden_size, self.config.hidden_size/self.config.head_num])
            a_sa2 = tf.get_variable('a_sa2', [self.config.head_num, 2*self.config.hidden_size/self.config.head_num, 1])


            Wr_wq = tf.get_variable('Wr_wq', [self.config.hidden_size, 1])
            Wm_wq = tf.get_variable('Wm_wq', [self.config.hidden_size, self.config.hidden_size])
            Wu_wq = tf.get_variable('Wu_wq', [self.config.hidden_size, self.config.hidden_size])

            Wr_wa = tf.get_variable('Wr_wa', [self.config.hidden_size, 1])
            Wm_wa = tf.get_variable('Wm_wa', [self.config.hidden_size, self.config.hidden_size])
            Wu_wa = tf.get_variable('Wu_wa', [self.config.hidden_size, self.config.hidden_size])         

            W_g1 = tf.get_variable('W_g1', [self.config.head_num, self.config.hidden_size, self.config.attention_size/self.config.head_num])
            a_g1 = tf.get_variable('a_g1', [self.config.head_num, 2*self.config.attention_size/self.config.head_num, 1])
           
            W_g2 = tf.get_variable('W_g2', [self.config.head_num, self.config.hidden_size, self.config.attention_size/self.config.head_num])
            a_g2 = tf.get_variable('a_g2', [self.config.head_num, 2*self.config.attention_size/self.config.head_num, 1])


            Wr = tf.get_variable('Wr', [self.config.attention_size, 1])
            Wm = tf.get_variable('Wm', [self.config.attention_size, self.config.attention_size])
            Wu = tf.get_variable('Wu', [self.config.attention_size, self.config.attention_size])
            #output weight
            W_l = tf.get_variable('W_l', [self.config.attention_size, self.config.class_num])
            b_l = tf.get_variable('b_l', [1, self.config.class_num])      

       # gat 1
        temp_Hq = []
        temp_Ha = []
        for i in range(self.config.head_num):
            temp_Hq.append(WordLevelGAT(Hq, self.input_question_graph, W_sq1[i], a_sq1[i], self.config.r))
            temp_Ha.append(WordLevelGAT(Ha, self.input_answer_graph, W_sa1[i], a_sa1[i], self.config.r))
        Hq = tf.concat(temp_Hq, -1)
        Ha = tf.concat(temp_Ha, -1)   #[num, len, atten]


       # gat 2
        temp_Hq = []
        temp_Ha = []
        for i in range(self.config.head_num):
            temp_Hq.append(WordLevelGAT(Hq, self.input_question_graph, W_sq2[i], a_sq2[i], self.config.r))
            temp_Ha.append(WordLevelGAT(Ha, self.input_answer_graph, W_sa2[i], a_sa2[i], self.config.r))
        Hq = tf.concat(temp_Hq, -1)
        Ha = tf.concat(temp_Ha, -1)   #[num, len, atten]

        

        Hq = SigmoidAtt(Hq, Wr_wq, Wm_wq, Wu_wq)
        Ha = SigmoidAtt(Ha, Wr_wa, Wm_wa, Wu_wa)
         
        count_question = 0
        count_answer = 0
        output_res = []
        for i in range(self.config.batch_size):
            temp_question = Hq[count_question:count_question+self.cur_question_len[i]]
            temp_answer = Ha[count_answer:count_answer+self.cur_answer_len[i]]
            
            all_nodes = tf.concat([temp_question, temp_answer], 0)

            #gat 1
            all_enc=[]
            for j in range(self.config.head_num):
                 all_enc.append(SenLevelGAT(all_nodes, W_g1[j], a_g1[j], self.config.r))
            
            temp_enc = tf.concat(all_enc, -1)
            
            #gat 2
            all_enc=[]
            for j in range(self.config.head_num):
                 all_enc.append(SenLevelGAT(temp_enc, W_g2[j], a_g2[j], self.config.r))

            temp_enc = tf.concat(all_enc, -1)

            temp_enc = tf.expand_dims(temp_enc, 0)
            temp_new = SigmoidAtt(temp_enc, Wr, Wm, Wu)
            if is_training:
               temp_new = tf.nn.dropout(temp_new, self.config.keep_prob)
            #output layer
            output_p = tf.add(tf.matmul(tf.tanh(temp_new), W_l), b_l)
            output_res.append(output_p)

            count_question += self.cur_question_len[i]
            count_answer += self.cur_answer_len[i]  
        
        output_res=tf.convert_to_tensor(output_res)
        ouput_label = tf.one_hot(self.input_label, self.config.class_num)
        self.prob = tf.nn.softmax(output_res)
      
#        print (tf.trainable_variables())
#        sys.exit()

        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = output_res, labels=ouput_label))
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer = tf.contrib.layers.l2_regularizer(0.0001), weights_list = [W_l, b_l])
        self.total_loss = self.loss+self.l2_loss



