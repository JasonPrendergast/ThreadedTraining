import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.contrib import rnn
import math
import sys
import threading, time, random

qfeatures=Queue.Queue(50)
qlabels=Queue.Queue(50)
finishedflag=0
errorcount=0
epoch_loss=0

hm_epochs = 1
n_classes = 2
batch_size = 1
chunk_size = 27001
n_chunks = 1
rnn_size = 128
total_batches = int(1600000/batch_size)


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')
layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}


class Producer:
    def __init__(self,food,lexicon):
        self.food=food#['ham','soup','salad']
        self.nextTime=0
        self.i=0
        self.lexicon=lexicon
        self.end=False
        
    def run(self):
        global qfeatures,qlabels,errorcount        
        while (self.i < len(self.food)):            
            batch_x = []
            batch_y = []
            batch_tweets=[]
            
            if(self.nextTime<time.clock()):
                if(self.i/100).is_integer():
                    print('insert'+str(self.i))
                line=self.food[self.i]
                label = line.split(':::')[0]
                tweet = line.split(':::')[1]
                #print(tweet)
                batch_tweets.append(tweet)
                current_words = word_tokenize(tweet.lower())
                current_words = [lemmatizer.lemmatize(i) for i in current_words]
                features = np.zeros(len(self.lexicon))
                for word in current_words:
                    if word.lower() in self.lexicon:
                        index_value = self.lexicon.index(word.lower())
                        features[index_value] += 1
                line_x = list(features)
                line_y = eval(label)
                if not all(v== 0 for v in line_x):
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    qfeatures.put(batch_x)
                    qlabels.put(batch_y)
                    self.nextTime+=(random.random())/100
                    self.i+=1
                else:
                    print('error')
                    self.nextTime+=(random.random())/100
                    self.i+=1
                    errorcount+=1

               
        print('finished in')

class Consumer:
    def __init__(self,maximum,sess,optimizer,cost):
        
        self.nextTime=1
        self.max=maximum
        self.i=0
        self.sess=sess
        self.optimizer=optimizer
        self.cost=cost
    def run(self):
        global qfeatures,finishedflag,qlabels,errorcount,epoch_loss
        while (self.i<(self.max-errorcount)):
            if(self.nextTime<time.clock() and not qfeatures.empty()):
                if(self.i/100).is_integer():
                    print('remove'+str(self.i))
                    print(epoch_loss)
                batch_x=qfeatures.get()
                batch_y=qlabels.get()
                #print('Removing')
                batch_x=np.array(batch_x)
                batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={x: batch_x,y: np.array(batch_y)})
                if math.isnan(c):
                    sys.exit('nan error')

                epoch_loss += c
                #print(epoch_loss)
                self.nextTime+=(random.random()*2)/100
                self.i+=1
                
        finishedflag=1
        print('finished out')




def recurrent_neural_network(x):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)
    #x = tf.split(0, n_chunks, x)

    #lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    #outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

saver = tf.train.Saver()
#saver = tf.train.import_meta_graph('./model.ckpt.meta')
tf_log = 'tf.log'
epoch_log ='epoch.log'

def train_neural_network(x):
    prediction = recurrent_neural_network(x)

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
        except:
            epoch = 1
            print('STARTING:',epoch)
        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,"./model.ckpt")
            epoch_loss = 1
            errorcount=0
            #epoch_loss=0
            food=[]
            with open('lexicon-2500-2638.pickle','rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                for line in f:
                    food.append(line)

            p=Producer(food,lexicon)
            c=Consumer(len(food),sess,optimizer,cost)
            pt=threading.Thread(target=p.run,args=())
            ct=threading.Thread(target=c.run,args=())
            pt.start()
            time.sleep(6)   
            ct.start()
            #time.sleep(10)
            while(finishedflag==0):
                time.sleep(1)
                #print('wait')   


            finishedflag==0           
            print('ending')
            saver.save(sess, "./model.ckpt")

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss,'nanerrors:',errorcount)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n')
            with open(epoch_log,'a') as f:
                f.write('Epoch'+str(epoch)+'completed out of'+str(hm_epochs)+'loss:'+str(epoch_loss)+'\n')
            epoch +=1

train_neural_network(x)
