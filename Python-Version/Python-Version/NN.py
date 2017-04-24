import tensorflow as tf 
import numpy as np
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
# First, read data from file and preprocess and other functions.
def Preprocess(FileName):
	Data = np.load(FileName)
	H_MBS = Data['H_MBS']
	H_SBS = Data['H_SBS']
	N_MS = Data['N_MS']
	N_MBS = Data['N_MBS']
	N_SBS = Data['N_SBS']
	N_frequency = Data['N_frequency']
	
	F = np.fft.fft(H_MBS,axis = 1)
	X = np.log(np.abs(F))
	Temp = X.flatten()
	Temp[np.isinf(Temp)] = -16
	#k_means or Lloyd
	Temp_sample = shuffle(Temp, random_state=0)[:3000]
	Temp_sample=Temp_sample.reshape(-1,1)
	kmeans=KMeans(n_clusters=20, random_state=0).fit(Temp_sample)
	Temp=Temp.reshape(-1,1)
	Temp = kmeans.predict(Temp)
	X=Temp.reshape(N_MS,N_MBS*N_frequency)
	#H_SBSr=np.reshape(H_SBS,N_MS*N_frequency,N_SBS)
	H_SBSr=H_SBS.reshape(N_MS*N_frequency,N_SBS)
	#H_SBSr and H_SBS are identical, so why should we add this line?
    
	y_temp = np.argmax(np.abs(H_SBSr),axis=1)
	y = np.zeros([N_MS,N_SBS])
	for i in range(N_MS):
	    y[i][y_temp[i]] = 1    
	return (X,y)

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Second, divide the dataset into Train and Test set
X,y = Preprocess('2000_100_5_1_DataCL.npz')#X: 2000*100, y:2000*1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)#divide the dataset
#Third, define a function to get minibatch from Train randomly
pass
#Fourth, tensorflow code here.
x = tf.placeholder(tf.float32, shape =[ None,np.shape(X_train)[1]])
y_label = tf.placeholder(tf.float32, shape = [None, np.shape(y)[1]])
layer1 = add_layer(x, np.shape(X_train)[1], 100, activation_function=tf.nn.relu)
y_result = add_layer(layer1, 100, np.shape(y)[1], activation_function=None)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_result))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

for i in range(np.shape(X_train)[0]/200):
	batch_x=X_train[i*200:i*200+199,:]
	batch_y=y_train[i*200:i*200+199]
	train_step.run(feed_dict={x: batch_x, y_label: batch_y})

correct_prediction = tf.equal(tf.argmax(y_result, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: X_test,y_label: y_test}))
