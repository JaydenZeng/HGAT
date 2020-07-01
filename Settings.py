class Config(object):
    def __init__(self):
        self.max_sub_question_len = 15
        self.max_sub_answer_len = 20
        self.keep_prob = 0.7
        self.hidden_size = 256
        self.lstm_hidden_size = 128
        self.batch_size = 32
        self.embedding_size = 256
        self.class_num = 4
        self.learning_rate = 0.001
        self.train_size = 8000
        self.test_size = 2000
        self.epoch_num = 10
        self.attention_size = 256
        self.head_num = 4
        self.r = 0.2
        self.domain ='shoe'
