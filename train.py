import tensorflow as tf 
import os
import sys
import datetime
import numpy as np
from Settings import Config
from Dataset import DataSet
from network import GATQA
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle

os.environ['CUDA_VISIBLE_DEVICES']='6,7'
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('train', True, 'set True to train')
conf_ = Config()
domain = conf_.domain
dataset = DataSet()
traindata = pickle.load(open('./data/'+domain+'/'+domain+'-train.pkl', 'rb'))
testdata  = pickle.load(open('./data/'+domain+'/'+domain+'-test.pkl', 'rb'))

def evaluation(y_pred, y_true):
    f1_s = f1_score(y_true, y_pred, average='macro')
    accuracy_s = accuracy_score(y_true, y_pred)
    return f1_s, accuracy_s


def train(sess, setting):
    with sess.as_default():
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('model', reuse = None, initializer = initializer):
            m = GATQA(is_training=FLAGS.train)
        global_step = tf.Variable(0, name='global_step', trainable = False)
        optimizer = tf.train.AdamOptimizer(setting.learning_rate)
        train_op = optimizer.minimize(m.total_loss, global_step=global_step)
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(max_to_keep=None)

        for epoch in range(setting.epoch_num):
            for i in range(int(setting.train_size/setting.batch_size)):
                nextQuestionBatch, nextAnswerBatch, nextLabelBatch, cur_len, nextQuestionGraphBatch,nextAnswerGraphBatch = dataset.nextBatch(traindata, testdata, FLAGS.train)
                feed_dict = {}
                feed_dict[m.input_question] = nextQuestionBatch
                feed_dict[m.input_answer] = nextAnswerBatch
                feed_dict[m.input_label] = nextLabelBatch
                feed_dict[m.cur_question_len] = cur_len['Q']
                feed_dict[m.cur_answer_len] = cur_len['A']
                feed_dict[m.input_question_graph] = nextQuestionGraphBatch
                feed_dict[m.input_answer_graph] = nextAnswerGraphBatch
                temp, step, loss_ = sess.run([train_op, global_step, m.loss], feed_dict)            

                if step%20==0:
                    time_str = datetime.datetime.now().isoformat()
                    tempstr = "{}: step {}, softmax_loss {:g}".format(time_str, step, loss_)
                    print (tempstr)
                    path = saver.save(sess, './model/MT_ATT_model', global_step=step)


def test(sess, setting):
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = GATQA(is_training=FLAGS.train)
        saver = tf.train.Saver()
        testlist = range(20, 5000, 20)
        best_model_iter = -1
        best_model_f1 = -1
        best_model_acc = -1

        for model_iter in testlist:
            try:
                saver.restore(sess, './model/MT_ATT_model-'+str(model_iter))
            except Exception:
                continue
            total_pred = []
            total_y = []
            for i in range(int(setting.test_size/setting.batch_size)):
                nextQuestionBatch, nextAnswerBatch, nextLabelBatch, cur_len, nextQuestionGraphBatch,nextAnswerGraphBatch = dataset.nextBatch(traindata, testdata, FLAGS.train)
                feed_dict = {}
                feed_dict[mtest.input_question] = nextQuestionBatch
                feed_dict[mtest.input_answer] = nextAnswerBatch
                feed_dict[mtest.input_label] = nextLabelBatch
                feed_dict[mtest.cur_question_len] = cur_len['Q']
                feed_dict[mtest.cur_answer_len] = cur_len['A']
                feed_dict[mtest.input_question_graph] = nextQuestionGraphBatch
                feed_dict[mtest.input_answer_graph] = nextAnswerGraphBatch               
                prob = sess.run([mtest.prob], feed_dict) #[1,32,4]
                for j in range(len(prob[0])):
                    total_pred.append(np.argmax(prob[0][j], -1))
                for item in nextLabelBatch:
                    total_y.append(item)

            with open('./temp_result.txt','w+') as fout:
                for i in range(len(total_pred)):
                    fout.write(str(total_y[i])+' '+str(total_pred[i])+'\n')
            f1,accuracy=evaluation(total_pred,total_y)
            if f1>best_model_f1:
                best_model_f1=f1
                best_model_acc=accuracy
                best_model_iter=model_iter
            print ('model_iter:',model_iter)
            print ('f1 score:',f1)
            print ('accuracy score:',accuracy)
        with open('./bestresult.txt','w+') as fout:
            fout.write('best_model_iter:'+str(best_model_iter)+'\n')
            fout.write('best_model_f1:'+str(best_model_f1)+'\n')
            fout.write('best_model_acc:'+str(best_model_acc))

        print ('----------------------------')
        print ('best model_iter', best_model_iter)
        print ('best f1 score: ', best_model_f1)
        print ('best accuracy score:', best_model_acc)
   
def main(_):
    #print (FLAGS.train)
    setting = Config()
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        if FLAGS.train == True:
            train(sess, setting)
        else:
            test(sess, setting)

if __name__ == '__main__':
    tf.app.run()
