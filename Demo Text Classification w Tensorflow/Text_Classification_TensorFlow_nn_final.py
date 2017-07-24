
# coding: utf-8
#    
#    TensorFlow is a versatile framework that can be used for a variety of Machine Learning problems 
#    besides Deep-Learning. 
#    This demo implements a classic Machine Learning problem of Text Classification with TensorFlow.
#    It uses a technique known as "bag of words" which relies on presence of certain words, their frequency 
#    and their uniqueness to form the features for learning. 
#    The classifier used is a "shallow" neural network with a single hidden layer.
#    
#    Author: Harshal Patil
#    
# In[ ]:


#     load packages
import os 
import zipfile
import urllib
import pandas as pd
import numpy as np
import string
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import time 
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import pickle


# In[ ]:


#     load the data
#    The data is received from http://jmcauley.ucsd.edu/data/amazon/ 
#    You will need to contact Julian McAuley of UCSD to obtain it

#    The data contains a sample of 1,000,000 Amazon product reviews 
#    There are -
#    Unique Review Writers: 267,504
#    Unique Products Reviewed: 615,419
#    Starting Date: Jan, 1997
#    Ending Date: Dec, 2014
  
#    I have extracted the data from Json structures and loaded into a dataframe
#    The dataframe will be read from the pickle
pname="amazon_reviews_df.pickle"
#fname = os.path.join(os.getcwd(),pname)
df = pd.read_pickle(pname)


# In[ ]:


noreviews = len(df)
users=len(df["reviewerID"].unique())
prods=len(df["asin"].unique())
df["datevec"]=pd.to_datetime(df["reviewTime"])
syear = min(df["datevec"].apply(lambda x: x.year))
smth = min(df["datevec"].apply(lambda x: x.month))
eyear = max(df["datevec"].apply(lambda x: x.year))
emth = max(df["datevec"].apply(lambda x: x.month))

print "No of Product Reviews, Unique Users, Unique Products, Year, Month To Year, Month"
print [noreviews, users, prods, syear, smth, eyear, emth]


# In[ ]:


#    We are interested in predicting a negative review purely based on review headline and review text
#    We consider any review that is a 2 star or 1 star as negative and label it as 1 

df["labelvec"] = np.vectorize(lambda x: 1 if(x<3.0) else 0)(df["overall"])
df["textvec"] = df["summary"] + " " + df["reviewText"]
 

#    check proper coding of label from class
print "1. check counts and percentages \n"
print pd.crosstab(df["labelvec"],df["overall"],margins=True)
print "\n"

#    check balance between positive and negative cases
#    check the percentage of positive cases 
print pd.crosstab(df["labelvec"],df["overall"],margins=True).apply(lambda x: (x/len(df))*100, axis=1)
print "\n"


# In[ ]:


#    preprocess strings to normalize and improve model performance
#    conver to lowercase, remove punctuations, numbers and extra spaces, tabs etc.
#    Stopwords removal at the time of Doc x Term matrix creation
if not os.path.exists("Amazondocterm.pickle"):
	t1 = time.time()
	message = np.vectorize(lambda x: x.lower())(df["textvec"])
	message = np.vectorize(lambda x: re.sub("["+string.punctuation+"]"," ",x))(message)
	message = np.vectorize(lambda x: re.sub("[0123456789]"," ",x))(message)
	message = np.vectorize(lambda x: " ".join(x.split()))(message)
	t2 = time.time()

	print "\n Text pre-processing time in seconds: " + str(t2-t1)


# In[ ]:


#    we will create a documentxterm matrix which will be passed to learner
#    The counts will be using TF-IDF to differentiate unique terms and add higher weight to them
#    you may need to run nltk.download()
#    consider only the top 5000 words as features

word_features = 5000

if not os.path.exists("Amazondocterm.pickle"):
	t1 = time.time()
	tfidfconvert = TfidfVectorizer(tokenizer=nltk.word_tokenize,stop_words="english",max_features=word_features)
	termdoc = tfidfconvert.fit_transform(message)
	t2=time.time()

	print "\n Doc x Term Matrix pre-processing time in seconds: " + str(t2-t1)
	with open("Amazondocterm.pickle","wb") as f:
		pickle.dump(termdoc,f)
else:
	with open("Amazondocterm.pickle","rb") as f:
		termdoc = pickle.load(f)

print "2. Document x Term Matrix size : "+ str(termdoc.shape[0])+" rows by "+ str(termdoc.shape[1])+" columns \n"


# In[ ]:


nrow = termdoc.shape[0]
ncol = termdoc.shape[1]

#    divide in training, cross validation and a test set 
train_set = np.random.choice(nrow,int(round(0.8*nrow)),replace=False)
remn_data = np.delete(range(nrow),train_set)
cv_set = np.random.choice(remn_data,int(round(0.9*remn_data.shape[0])),replace=False)
test_set = np.setdiff1d(remn_data,cv_set)

print "4. Training Set Size : " + str(len(train_set))
print "5. Cross-Validation Set Size : " + str(len(cv_set))
print "6. Test Set Size : " + str(len(test_set))

# actual training and test data
matrix_train = termdoc[train_set] 
matrix_cv = termdoc[cv_set] 
matrix_test = termdoc[test_set] 
target_train = np.transpose([df["labelvec"][train_set]])
target_cv = np.transpose([df["labelvec"][cv_set]])
target_test = np.transpose([df["labelvec"][test_set]])


# In[ ]:


print "\n\n Establish TensorFlow graph \n"
#    build tensorflow graph
input_layer = word_features
hidden_layer = 50 #arbitrary no of units in hidden layer
output_layer = 1

# DEPLOYMENT CHOICE with Simple Settings
# specify device - we will change this to GPU to deploy on specific GPU
with tf.device('/gpu:0'):
#with tf.device('/cpu:0'):
    
    with tf.name_scope('input') as scope:
        inp = tf.placeholder(tf.float32, [None, input_layer],name="inp")
        hidden_w = tf.Variable(tf.random_normal(shape=[input_layer,hidden_layer]),name="hidden_w")
        hidden_b = tf.Variable(tf.random_normal(shape=[hidden_layer]),name="hidden_b")
        inp_model = tf.add(tf.matmul(inp,hidden_w),hidden_b)
        
    #init all weights to random
    with tf.name_scope('hidden') as scope:
        #hidden_model = tf.nn.relu(inp_model)
        hidden_model = tf.sigmoid(inp_model)
    
    with tf.name_scope('output') as scope:
        op = tf.placeholder(tf.float32, [None, output_layer],name="op")
        op_w = tf.Variable(tf.random_normal(shape=[hidden_layer,output_layer]),name="op_w")
        op_b = tf.Variable(tf.random_normal(shape=[output_layer]),name="op_b")
        op_model = tf.add(tf.matmul(hidden_model,op_w),op_b)
        predict = tf.round(tf.sigmoid(op_model))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,op), tf.float32))

    with tf.name_scope('cost') as scope:
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=op_model,labels=op))
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0025).minimize(cost)
    


# In[ ]:


init = tf.global_variables_initializer()

# LOG DEVICE PLACEMENTs - show that the graph is deployed to CPU or GPU
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#sess = tf.Session()

# create initialized variables
sess.run(init)


# In[ ]:


#    training the model - we go through a number of random batches to train
batch=50000
rounds = 1000
# record at these iterations and print
recordfreq = range(0,10,1) + range(10,100,10) + range(100,(rounds+100),100)
printfreq = 100

# test accuracy in training loop is calculated on cross-validation 
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
runtime = []
i=0

time_start = time.time()
for i in range(rounds):
#        Train
    rand_index = np.random.choice(matrix_train.shape[0], size=batch)
    rand_x = matrix_train[rand_index].todense() 
    rand_y = target_train[rand_index]
    sess.run(optimizer, feed_dict={inp: rand_x, op: rand_y})
    
    # Only record loss and accuracy every 100 generations
    if (i) in recordfreq:
        i_data.append(i+1)
        t1 = time.time()
        train_loss_temp = sess.run(cost, feed_dict={inp: rand_x, op: rand_y})
        t2=time.time()
        train_loss.append(train_loss_temp)
        runtime.append(t2-t1)
        
        t1 = time.time()
        test_loss_temp = sess.run(cost, feed_dict={inp: matrix_cv.todense(), op: target_cv})
        t2=time.time()
        test_loss.append(test_loss_temp)
        runtime.append(t2-t1)
         
        t1 = time.time()
        train_acc_temp = sess.run(accuracy, feed_dict={inp: rand_x, op: rand_y})
        t2=time.time()
        train_acc.append(train_acc_temp)
        runtime.append(t2-t1)
        
        t1 = time.time()
        test_acc_temp = sess.run(accuracy, feed_dict={inp: matrix_cv.todense(), op: target_cv})
        t2=time.time()
        test_acc.append(test_acc_temp)
        runtime.append(t2-t1)
    
    if (i) in recordfreq:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Validation Loss): {:.2f} ({:.2f}). Train Acc (Validation Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

time_end = time.time()
print "\n 6. Training time in seconds (total) " + str(time_end - time_start)


# In[ ]:


# write the graph to a log file to see via Tensorboard
writer = tf.summary.FileWriter('tflogs', sess.graph)
print sess.run(accuracy,feed_dict={inp: matrix_test.todense(), op: target_test})
writer.close()


# In[ ]:


print "\n 8. Loss Function"
plt.plot(i_data,train_loss,label="Training Set")
plt.plot(i_data,test_loss,label="Validation Set")
plt.legend()
plt.show()


# In[ ]:


print "9. Accuracy"
plt.plot(i_data,train_acc,label="Training Set")
plt.plot(i_data,test_acc,label="Validation Set")
plt.legend()
plt.show()


# In[ ]:


print "10. Final Model Evaluation on Test Set"

predictions  = np.vectorize(lambda x: 1 if x>0.5 else 0)(sess.run(predict, feed_dict={inp: matrix_test.todense(), op: target_test}))

testdf = pd.DataFrame(predictions,columns=["predictions"])
testdf["target"] = target_test
posdf = testdf[testdf["target"]==1]

true_pos = float(sum(posdf["predictions"]))
tot_pred = sum(testdf["predictions"])
all_pos = len(posdf)

prec = round(true_pos/tot_pred,2)
rec = round(true_pos/all_pos,2)
fscore = 2*((prec*rec)/(prec+rec))

print "Accuracy from TensorFlow : " + str(sess.run(accuracy, feed_dict={inp: matrix_test.todense(), op: target_test}))
print "Precision %: " + str(prec*100)
print "Recall %: " + str(rec*100) 
print "F-Score: (max is 1) " + str(fscore)


# In[ ]:




