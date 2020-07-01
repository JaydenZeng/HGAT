# -*- coding: UTF-8 -*-
from Settings import Config
import re
import os
import sys
import numpy as np
import pickle



class DataSet:
    def __init__(self):
        self.config = Config()
        self.domain = self.config.domain
        self.iter_num = 0
        self.label_set = ['neutral', 'conflict', 'negative', 'positive']
        self.question = np.load('./data/'+self.domain+'/'+self.domain+'-que-15.npy')
        self.answer = np.load('./data/'+self.domain+'/'+self.domain+'-ans-20.npy')
        self.question_graph = np.load('./data/'+self.domain+'/'+self.domain+'-que-allgraph.npy')
        self.answer_graph = np.load('./data/'+self.domain+'/'+self.domain+'-ans-allgraph.npy')



    def cal_length(self, filepath):
        all_length = []
        with open(filepath, 'r+') as fr:
            lines =fr.readlines()
            for line in lines:
                count = 0
                tmp_seg = line.split(' ')
                for seg in tmp_seg:
                    if seg != '' and seg != '\n':
                        count += 1
                all_length.append(count)
        return all_length

    def setdata(self, question_len, answer_len, domain_index):
        alldata = {'Q':[], 'A':[], 'L':[], 'QG':[], 'AG':[]}
        count = 0
        for i in range(len(question_length)):
            tmp = []
            tmp_graph = []
            for j in range(count, count+question_length[i]):
                tmp.append(self.question[j])
                tmp_graph.append(self.question_graph[j])
            alldata['Q'].append(tmp)
            alldata['QG'].append(tmp_graph)
            count += question_length[i]

        count = 0
        for i in range(len(answer_length)):
            tmp = []
            tmp_graph = []
            for j in range(count, count+answer_length[i]):
                tmp.append(self.answer[j])
                tmp_graph.append(self.answer_graph[j])
            alldata['A'].append(tmp)
            alldata['AG'].append(tmp_graph)
            
            count += answer_length[i]

        with open('./data/label.txt', 'r+') as fr:
            lines = fr.readlines()
            for line in lines[domain_index:domain_index+10000]:
                alldata['L'].append(self.label_set.index(line.strip()))
        return alldata  

    def set_train_test_data(self, data, train_size):
        traindata = {'Q':[], 'A':[], 'L':[], 'QG':[], 'AG':[]}
        testdata = {'Q':[], 'A':[], 'L':[], 'QG':[], 'AG':[]}
        temp_order = list(range(len(data['Q'])))
        np.random.shuffle(temp_order)
        for i in range(len(data['Q'])):
            if i < train_size:
                traindata['Q'].append(data['Q'][temp_order[i]])
                traindata['A'].append(data['A'][temp_order[i]])
                traindata['L'].append(data['L'][temp_order[i]])
                traindata['QG'].append(data['QG'][temp_order[i]])
                traindata['AG'].append(data['AG'][temp_order[i]])
            else:
                testdata['Q'].append(data['Q'][temp_order[i]])
                testdata['A'].append(data['A'][temp_order[i]])
                testdata['L'].append(data['L'][temp_order[i]])
                testdata['QG'].append(data['QG'][temp_order[i]])
                testdata['AG'].append(data['AG'][temp_order[i]])
        return traindata, testdata


    def nextBatch(self, traindata, testdata, is_training=True):
        nextQuestionBatch=[]
        nextAnswerBatch=[]
        nextLabelBatch=[]
        cur_len = {'Q':[], 'A':[]}

        if is_training:
            if (self.iter_num+1)*self.config.batch_size > len(traindata['Q']):
                self.iter_num = 0
            if self.iter_num == 0:
               self.temp_order = list(range(len(traindata['Q'])))
               np.random.shuffle(self.temp_order)
            temp_order = self.temp_order[self.iter_num*self.config.batch_size:(self.iter_num+1)*self.config.batch_size]
        else:
            if (self.iter_num+1)*self.config.batch_size > len(testdata['Q']):
                self.iter_num = 0
            if self.iter_num == 0:
                self.temp_order = list(range(len(testdata['Q'])))
                #np.random.shuffle(self.temp_order)
            temp_order = self.temp_order[self.iter_num*self.config.batch_size:(self.iter_num+1)*self.config.batch_size]
        
        question = []
        answer = []
        label = []
        question_graph = []
        answer_graph = []
        for it in temp_order:
            if is_training:
                for item in traindata['Q'][it]:
                    question.append(item)
                for item in traindata['A'][it]:
                    answer.append(item)
                for item in traindata['QG'][it]:
                    question_graph.append(item)
                for item in traindata['AG'][it]:
                    answer_graph.append(item)
                label.append(traindata['L'][it])
                cur_len['Q'].append(len(traindata['Q'][it]))
                cur_len['A'].append(len(traindata['A'][it]))
            else:
                for item in testdata['Q'][it]:
                    question.append(item)
                for item in testdata['A'][it]:
                    answer.append(item)
                for item in testdata['QG'][it]:
                    question_graph.append(item)
                for item in testdata['AG'][it]:
                    answer_graph.append(item)
                label.append(testdata['L'][it])
                cur_len['Q'].append(len(testdata['Q'][it]))
                cur_len['A'].append(len(testdata['A'][it]))
        self.iter_num += 1

        nextQuestionBatch = np.array(question)
        nextAnswerBatch = np.array(answer)
        nextLabelBatch = np.array(label)
        nextQuestionGraphBatch = np.array(question_graph)
        nextAnswerGraphBatch = np.array(answer_graph)

        return nextQuestionBatch, nextAnswerBatch, nextLabelBatch, cur_len, nextQuestionGraphBatch, nextAnswerGraphBatch


if __name__ == '__main__':
    data = DataSet()
    print (np.shape(data.question))
    print (np.shape(data.question_graph), np.shape(data.answer_graph))
   # sys.exit()
    domain_index= 0
    if data.domain == 'beauty':
       domain_index = 0
    elif data.domain == 'electronic':
       domain_index = 10000
    else:
       domain_index = 20000
    print ('cur_domain:', data.domain)
    print ('current domain index:', domain_index)

    question_length = data.cal_length('./data/'+data.domain+'/'+data.domain+'-que.split')
    answer_length = data.cal_length('./data/'+data.domain+'/'+data.domain+'-ans.split')

    alldata = data.setdata(question_length, answer_length, 0)
    print (len(alldata['Q']),len(alldata['A']),len(alldata['L']))
     
    traindata, testdata = data.set_train_test_data(alldata, 8000)
    print (len(traindata['Q']), traindata['Q'][0], np.shape(traindata['Q'][0]))

    pickle.dump(traindata, open('./data/'+data.domain+'/'+data.domain+'-train.pkl', 'wb'))
    pickle.dump(testdata, open('./data/'+data.domain+'/'+data.domain+'-test.pkl','wb'))

    nextQuestionBatch, nextAnswerBatch, nextLabelBatch, cur_len, nextQuestionGraphBatch,nextAnswerGraphBatch = data.nextBatch(traindata, testdata, True)
    print (np.shape(nextQuestionBatch),np.shape(nextAnswerBatch))
    print (cur_len, sum(cur_len['Q']))
    print (nextLabelBatch)

