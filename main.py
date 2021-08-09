import tensorflow as tf
import tensorflow_datasets as datasets
# FOR 1.14V: from tensorflow.examples.tutorials.mnist import input_data #MNIST numbers for classification
# Construct a tf.data.Dataset

#Read
# FOR 1.4V: mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) #one hot encoding to get the number
mnist = datasets.load(name="mnist", split=tfds.Split.TRAIN)

#Parameters

n_train = mnist.train.num_examples #55.000 
n_validation = mnist.validation.num_examples #5.000
n_test = mnist.test.num_examples #10.000
 

#NN Architecture

n_input = 784 #input Layer (28x28 pixels)
n_hidden1 = 512 #1st hidden layer
n_hidden2 = 256 #2nd hidden layer
n_hidden3 = 128 #3rd hidden layer
n_output = 10 #output layer (0-9 digits)

#Parameters

learning_rate = 1e-4 # How much the parameters will adjunst each step of the learning process
n_iterations = 1000 # Cuantas iteraciones por cada paso
batch_size = 128 # Cuantos training examples se ejecutan en cada paso.
dropout = 0.5 # Probability to be deleted 

#TensorFlow Graph
X = tf.placeholder("float", [None,n_input])
Y = tf.placeholder("float", [None,n_output])
keep_prob = tf.placeholder(tf.float32) #Control de dropout rate

#Los valores a ser optimizados durante el entrenamiento comenzaran seteados en cero (Estos valores tienen un fuerte impacto en el accuracy del modelo)

#Neural Net

weights = {'w1' : tf.Variable(tf.truncated_normal([n_input,n_hidden1], stddev=0.1)), 
           'w2': tf.Variable(tf.truncated_normal([n_hidden1,n_hidden2], stddev=0.1)),
           'w3': tf.Variable(tf.truncated_normal([n_hidden2,n_hidden3], stddev=0.1)),
           'out': tf.Variable(tf.truncated_normal([n_hidden3,n_output], stddev=0.1)),}

#BIAS

biases = {
    'b1': tf.Variable(tf.constant(0.1,shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1,shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1,shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1,shape=[n_output])) 
}

#LAYERS

layer_1 = tf.add(tf.matmul(X, weights['w1']) , biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']) , biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']) , biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))

##Neural Net

correct_pred = tf.equal(tf.argmax(output_layer, tf.argmax(Y,1)))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Train mini batches

for i in range(n_interations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dic={
        X: batch_x , Y: batch_y, keep_prob:dropout
    })

#Print loss and accuracy (per batch)
if i %100 == 0:
    minibatch_loss, minibatch_accuracy = sess.run(
        [cross_entropy, accuracy], 
        feed_dict={X:batch_x, Y:batch_y, keep_prob:1.0}
    )

print(
    "Iteration",
    str(i),
    "\t| Loss=",
    str(minibatch_loss),
    "\t| Accuracy =",
    str(minibatch_accuracy)
    )

test_accuracy = sess.run(accuracy,feed_dict={X:mnist.test.images, Y:
mnist.test.labels, keep_prob:1.0})
print("\nAccuracy on test set:", test_accuracy)

