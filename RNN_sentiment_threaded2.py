import tensorflow as tf
import threading, time, random
import queue as Queue
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import math
import sys
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

qfeatures=Queue.Queue(50)
qlabels=Queue.Queue(50)
finishedflag=0
errorcount=0
epoch_loss=0
errorbatch=0

n_nodes_hl1 = 500
n_nodes_hl2 = 500

hm_epochs = 1
n_classes = 2
batch_size = 32
chunk_size = 27001
n_chunks = 1
rnn_size = 128
total_batches = int(1600000/batch_size)


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

#layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
#             'biases':tf.Variable(tf.random_normal([n_classes]))}
hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([rnn_size, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


class Producer:
    def __init__(self,food,lexicon):
        self.food=food#['ham','soup','salad']
        self.nextTime=0
        self.i=0
        self.lexicon=lexicon
        self.end=False
        #self.batch_x = []
        #self.batch_y =[]
        #self.batch_tweets=[]
        #batches_run=0
    def run(self):
        global qfeatures,qlabels,errorcount        
        while (self.i < len(self.food)):
            if (self.i/100).is_integer():
                    print('inserting: '+str(self.i))
            batch_x = []
            batch_y = []
            batch_tweets=[]
            #batches_run = 0
            #print('inwhile')
            #print(self.i)
            if(self.nextTime<time.clock()):
                #for b in range(batch_size):
                line=self.food[self.i]#random.randrange(len(self.food))]
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
                    if errorcount==32:
                        errorbatch+=1
                        errorcount=0
                    
                
                    
                    
                #print(' Current '+str(self.i)+' '+str(len(self.food)))
                #'qfeatures '+str(batch_x)+' labels '+str(batch_y)+
                
            #self.end=True
        print('finished in')

class Consumer:
    def __init__(self,maximum,sess,optimizer,cost):
        #self.food=['ham','soup','salad']
        self.nextTime=1
        self.max=(maximum/batch_size)
        self.i=0
        self.sess=sess
        self.optimizer=optimizer
        self.cost=cost
    def run(self):
        global qfeatures,finishedflag,qlabels,errorcount,epoch_loss
        batch_x=[]
        batch_y=[]
        while (self.i<(self.max-(errorbatch+1))):
           #print(self.i)
           # print(self.max)
            #print(errorcount)
            if(self.nextTime<time.clock() and not qfeatures.empty()):
                while (len(batch_x)<batch_size):
                    #print()
                    batch_x.append(qfeatures.get())
                    batch_y.append(qlabels.get())
                
                batch_x=np.array(batch_x)
                #print('batch length')
                #print(len(batch_x))
                batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))
                #print('after2')
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={x: batch_x,y: np.array(batch_y)})
                #print('after3')
                if not math.isnan(c):
                    print('not NAN??')
                    print(c)
                    #c=1
                #    sys.exit('nan error')

                epoch_loss += c
                #print(c)
                if (self.i/100).is_integer():
                    print('Removing: '+str(self.i)+'cost: '+str(epoch_loss))
                #print('after4')
                #print(epoch_loss)
                #print('after5')
                self.nextTime+=(random.random()*2)/100
                #print(self.nextTime)
                self.i+=1
                #print('after7')
                #f=self.food[random.range(len(self.food))]
                #q.put(f)
               # print('adding'+f)
                #self.nextTime+=random.random()
        finishedflag=1
        print('finished out')




def recurrent_neural_network(x):
    

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)
    #x = tf.split(0, n_chunks, x)

    #lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    
    l1 = tf.add(tf.matmul(x,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.tanh(l2)#relu(l2)
    
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],output_layer['weight']) + output_layer['bias']

    return output

saver = tf.train.Saver()
#saver = tf.train.import_meta_graph('./model.ckpt.meta')
tf_log = 'tf.log'
epoch_log ='epoch.log'

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    
    
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
            errorbatch=0
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
            time.sleep(4)   
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
        #try:
        
        #except:
        #    print('didnt save')
        #    sys.exit('save error')
        

        #for epoch in range(hm_epochs):
        #    epoch_loss = 0
        #    for _ in range(int(mnist.train.num_examples/batch_size)):
        #        epoch_x, epoch_y = mnist.train.next_batch(batch_size)
        #        epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
#
        #        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
        #        epoch_loss += c

        #    print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

       # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

       # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
       # print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)
