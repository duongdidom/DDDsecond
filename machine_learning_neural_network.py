"""
NEURAL NETWORK

deep neural network when there is more than 1 hidden layer
All machine learning above are good at classification, not logic. Neural network is good at logic

TensorFlow is not pure python code, they've only basically created a nice pretty Python wrapper to make our lives easier. TensorFlow inherently runs on a computation graph. The graph is first described by neural_network_model in our code, but very quickly it's iterated on, based on the rules we set forth in the train_neural_network function... but every iteration in train_neural_network does not reset the values, it's working on the computation graph. This abstraction is notably confusing, but that's what's happening, and that abstraction is made in the name of simplicity and allowing us mere mortal Python developers make use of the technology.

Data source: Image data = imagenet. Text data = wikipedia data dump. Chat = chat log. Speech = tatoeba
""" """
input > weight > hidden layer 1 > run through activation function > weight > hidden layer 2
> run through activation function > weight > output layer

compare output vs actual output > cost function (e.g. cross entropy)
optimisation function (e.g. optimiser) > minimise cost (e.g. AdamOptimizer ... SGD, AdaGrad)

backpropagation to backward manipulating the weight

feed forward + backprop = epoch
""" """
1. small sample: recognize digit number from 0-9
    |   ├── create model
    |   ├── train
2. catergorize positive and negative tone: given data from 2 text files. 
    ├── create training set and testing set
    |   ├── create lexicon: set of common words
    |   ├── sample handling: For each classification, use result from lexicon, if any word match result in lexicon, assign 1 to the list.
    |   ├── create model
    |   ├── train
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random, pickle
from collections import Counter

"""
train and recognise digit number 0-9
"""


def NeuralNetwork_sample():
    # create neural net work model
    def neural_network_model(data):
        # define nodes for each hiden layer. Doesn't have to be the same for each layer, but this case we do for testing purpose
        n_nodes_hidden_layer1 = 500
        n_nodes_hidden_layer2 = 500
        n_nodes_hidden_layer3 = 500

        # number of classes
        n_classes = 10

        # weight = tensorflow variable. Out of which, = randomise normal create the shape that we want: 784 * number of nodes in hidden layer 1
        ### Where input data is 0, biases ensure that the value coming through function is not 0, ie will always be on
        hidden_layer_1 = {
            "weights": tf.Variable(tf.random_normal([784, n_nodes_hidden_layer1])),
            "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer1])),
        }

        # weight in hidden layer 2 has the shape of: number of output nodes from hidden layer 1 * number of output nodes in hidden layer 2
        hidden_layer_2 = {
            "weights": tf.Variable(
                tf.random_normal([n_nodes_hidden_layer1, n_nodes_hidden_layer2])
            ),
            "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer2])),
        }

        hidden_layer_3 = {
            "weights": tf.Variable(
                tf.random_normal([n_nodes_hidden_layer2, n_nodes_hidden_layer3])
            ),
            "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer3])),
        }

        output_layer = {
            "weights": tf.Variable(
                tf.random_normal([n_nodes_hidden_layer3, n_classes])
            ),
            "biases": tf.Variable(tf.random_normal([n_classes])),
        }

        # (input data * weight) + biases.
        # layer 1
        l1 = tf.add(
            tf.matmul(data, hidden_layer_1["weights"]), hidden_layer_1["biases"]
        )

        # activation function = rectify function ~ threshold function
        l1 = tf.nn.relu(l1)

        # layer 2
        l2 = tf.add(tf.matmul(l1, hidden_layer_2["weights"]), hidden_layer_2["biases"])

        # activation function = rectify function ~ threshold function
        l2 = tf.nn.relu(l2)

        # layer 3
        l3 = tf.add(tf.matmul(l2, hidden_layer_3["weights"]), hidden_layer_3["biases"])

        # activation function = rectify function ~ threshold function
        l3 = tf.nn.relu(l3)

        # output layer
        output = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]

        return output

    # run the neural network model
    def train_neural_network(actual_feature, actual_label):
        # how many picture will be computed at each time
        batch_size = 100

        # run model to retrieve prediction
        prediction = neural_network_model(actual_feature)

        # compute cost. Calculate difference between prediction we got and the actual label that we have
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction, labels=actual_label
            )
        )

        # minimise the cost
        optimiser = tf.train.AdamOptimizer().minimize(
            cost
        )  # default learning rate of Adam Optimizer = 0.001

        # set number of cycles feed forward + backdrop fixing the weights
        n_epochs = 10

        # open tensorflow session
        with tf.Session() as sess:
            # initiate session
            sess.run(tf.global_variables_initializer())

            # train the network. loop through all epochs
            for epoch in range(n_epochs):
                # first assume that there is not loss
                epoch_loss = 0

                # loop through each batch size in sample data
                for _ in range(int(mnist.train.num_examples / batch_size)):
                    # trunch through data set
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                    # feed for each x and y, optimize the cost
                    _, c = sess.run(
                        [optimiser, cost], feed_dict={x: epoch_x, y: epoch_y}
                    )

                    # trach epoch loss
                    epoch_loss += c

                print("Epoch", epoch+1, "complete out of ", n_epochs, "loss:", epoch_loss)

            # once training is complete.
            # Checking if predction and actual label y are identical:
            # argmax returns index of maximum value. Equal() function check whether the index location are the same
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(actual_label, 1))

            # calculate accuracy = convert to float correct value
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))

            # evaluate the accuracy of test images and test labels
            print(
                "Accuracy = ",
                accuracy.eval({x: mnist.test.images, y: mnist.test.labels}),
            )

    # import data to recognise digits. One_hot = one component will be hot, equivalent to being 1 in either 0 or 1
    ### if we have 3 classes, 0-2, we would normally classify them to 3 variables such as 0=0, 1=1, 2=2.
    ### instead one_hot classify them as 0 = [1,0,0], 1=[0,1,0], 2=[0,0,1]
    mnist = input_data.read_data_sets(r"/Datamart", one_hot=True)

    # placeholder. Shape = height x width. 784 because each image is size 28 x 28, so there are 784 pixels in total
    x = tf.placeholder("float", shape=[None, 784])
    y = tf.placeholder("float")

    # call function train neural network
    train_neural_network(x, y)


"""
train and categorise positive and negative tone of text
each word that we feed into machine has to be exactly same length and shape
"""


# create training set and testing set. Test size proportion = the size of total data size that will be assigned for testing
def create_features_set_and_labels(pos, neg, test_size_proportion=0.1):
    # lemmatizer find similar word from given word
    ### e.g. print(lemmatizer.lemmatize("cats"));print(lemmatizer.lemmatize("rocks")); print(lemmatizer.lemmatize("better", pos="a"))
    lemmatizer = WordNetLemmatizer()

    # how many lines will be run each run
    hm_lines = 10000000
    
    # create lexicon = list of common word found in positive and negative files
    def create_lexicon(pos, neg):
        # start with empty set that we will store unique words
        lexicon = []

        # open each positive and negative file
        for _file in [pos, neg]:
            # open file in read mode
            with open(_file, "r") as f:
                # read each line and store in contents variable
                contents = f.readlines()

                # loop each line in contents for up to however many lines we set above
                for l in contents[:hm_lines]:
                    # use word tokenize to seperate each word in line into an item in a list
                    all_words = word_tokenize(l.lower())

                    # add each word found into lexicon
                    lexicon += list(all_words)

        # lemmatize each word in lexicon
        lexicon = [lemmatizer.lemmatize(w) for w in lexicon]

        # count words and store in dictionary. Each word = key. Value for each key = how many time each word appear
        ### e.g. {'the': 123123, 'a':64545, 'hungry':123,...}
        word_count = Counter(lexicon)

        # create empty list
        final_lexicon = []

        # loop through every key in word count
        for w in word_count:
            # if the word count of word w is between 50 and 1000
            ### we assume that any word more than 1000 is too common such as 'a', 'an', 'the', 'and'
            ### we assume that any word less than 50 is too rare
            if 50 < word_count[w] < 1000:
                # append it to final lexicon
                final_lexicon.append(w)

        return final_lexicon

    # handle sample. Given sample file, lexicon, classification
    def sample_handling(sample, lexicon, classification):
        # set empty feature set list
        feature_set = []

        # open sample file in read mode
        with open(sample, "r") as f:
            contents = f.readlines()
            # loop through every line
            for l in contents[:hm_lines]:
                # for line, convert to lower case, then seperate each word to item in a list
                current_word = word_tokenize(l.lower())

                # lemmatize each word = find similar meaning word. Then insert them back into current word list
                current_word = [lemmatizer.lemmatize(w) for w in current_word]

                # count length of lexicon. Create a list of however many zero the length of lexicon has
                features = np.zeros(len(lexicon))

                # loop each word in current word list
                for w in current_word:
                    # if the word is in lexicon list
                    if w.lower() in lexicon:
                        # find the index number in list lexicon
                        index_value = lexicon.index(w.lower())

                        # set value in features list from 0 to 1
                        features[index_value] = 1

                # create a list out of current feature list
                features = list(features)

                # append feature list to feature set list, along with our given (known) classification
                # feature_set will look like    [
                #                               [[0010101], [10] if positive or [01] if negative]
                #                               [[feature],[classification]]
                #                               [[]]
                #                               ]
                feature_set.append([features, classification])

        return feature_set
        
    # return found lexicon from above custom lexicon function, out of given positive and negative txt file
    lexicon = create_lexicon(pos, neg)
    print("lexicon size = ", len(lexicon))

    # create empty feature list
    features = []

    # append to feature list feature_set of each sample file, given found lexicon above and known classification
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])

    # suffle the data to create randomness
    random.shuffle(features)

    # create array out of feature
    features = np.array(features)

    # test size = all data size * test size proportion
    test_size = int(test_size_proportion * len(features))

    # create training feature out of feature list
    # feature list is constructed as [ [[feature], [classification]]
    #                                  [[feature], [classification]] ]
    # [:,0] will extract only the first element in each list. We extract from feature list --> so this function will extract only the feature
    # extract it up to before the test size
    train_x = list(features[:, 0][:-test_size])

    # create training label
    train_y = list(features[:, 1][:-test_size])

    # create testing feature
    test_x = list(features[:, 0][-test_size:])

    # create testing label
    test_y = list(features[:, 1][-test_size:])

    return train_x, train_y, test_x, test_y


# create model for our positive/ negative tone sample. Similar to neural model sample created above
def neural_network_model(data):
    # define nodes for each hiden layer. Doesn't have to be the same for each layer, but this case we do for testing purpose
    n_nodes_hidden_layer1 = 1500
    n_nodes_hidden_layer2 = 1500
    n_nodes_hidden_layer3 = 1500

    # number of classes
    n_classes = 2

    # weight = tensorflow variable. Out of which, = randomise normal create the shape that we want: 784 * number of nodes in hidden layer 1
    ### Where input data is 0, biases ensure that the value coming through function is not 0, ie will always be on
    hidden_layer_1 = {
        "weights": tf.Variable(tf.random_normal([len(data[0]), n_nodes_hidden_layer1])),
        "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer1])),
    }

    # weight in hidden layer 2 has the shape of: number of output nodes from hidden layer 1 * number of output nodes in hidden layer 2
    hidden_layer_2 = {
        "weights": tf.Variable(
            tf.random_normal([n_nodes_hidden_layer1, n_nodes_hidden_layer2])
        ),
        "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer2])),
    }

    hidden_layer_3 = {
        "weights": tf.Variable(
            tf.random_normal([n_nodes_hidden_layer2, n_nodes_hidden_layer3])
        ),
        "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer3])),
    }

    output_layer = {
        "weights": tf.Variable(
            tf.random_normal([n_nodes_hidden_layer3, n_classes])
        ),
        "biases": tf.Variable(tf.random_normal([n_classes])),
    }

    # (input data * weight) + biases.
    # layer 1
    l1 = tf.add(
        tf.matmul(data, hidden_layer_1["weights"]), hidden_layer_1["biases"]
    )

    # activation function = rectify function ~ threshold function
    l1 = tf.nn.relu(l1)

    # layer 2
    l2 = tf.add(tf.matmul(l1, hidden_layer_2["weights"]), hidden_layer_2["biases"])

    # activation function = rectify function ~ threshold function
    l2 = tf.nn.relu(l2)

    # layer 3
    l3 = tf.add(tf.matmul(l2, hidden_layer_3["weights"]), hidden_layer_3["biases"])

    # activation function = rectify function ~ threshold function
    l3 = tf.nn.relu(l3)

    # output layer
    output = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]

    return output


# run the neural network model
def train_neural_network(train_feature, train_label, test_feature, test_label):
    # how many picture will be computed at each time
    batch_size = 100

    # run model to retrieve prediction
    prediction = neural_network_model(train_feature)

    # compute cost. Calculate difference between prediction we got and the actual label that we have
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction, labels=train_label
        )
    )

    # minimise the cost
    optimiser = tf.train.AdamOptimizer().minimize(
        cost
    )  # default learning rate of Adam Optimizer = 0.001

    # set number of cycles feed forward + backdrop fixing the weights
    n_epochs = 10

    # open tensorflow session
    with tf.Session() as sess:
        # initiate session
        sess.run(tf.global_variables_initializer())

        # train the network. loop through all epochs
        for epoch in range(n_epochs):
            # first assume that there is not loss
            epoch_loss = 0

            # set default value for i to loop for each batch size in actual feature 
            i = 0 
            while i < len(train_feature):
                # start, end value
                start = i
                end = i + batch_size

                # slice train feature and train label, where size = batch size 
                batch_x = np.array(train_feature[start:end])
                batch_y = np.array(train_label[start:end])

                # feed for each x and y, optimize the cost
                _, c = sess.run(
                    [optimiser, cost], feed_dict={x: epoch_x, y: epoch_y}
                )

                # trach epoch loss
                epoch_loss += c

                # add batch size to i
                i += batch_size

            print("Epoch", epoch+1, "complete out of ", n_epochs, "loss:", epoch_loss)

        # once training is complete.
        # Checking if predction and actual label y are identical:
        # argmax returns index of maximum value. Equal() function check whether the index location are the same
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(train_label, 1))

        # calculate accuracy = convert to float correct value
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))

        # evaluate the accuracy of test images and test labels
        print('Accuracy:', accuracy.eval({x:test_feature, y:test_label}))


# call function to create training set and testing set
train_x, train_y, test_x, test_y = create_features_set_and_labels(r"Datamart\positive.txt", r"Datamart\negative.txt")

# # (optional) store these training and testing set into binary pickle file
# with open(r"Datamart\sentiment_set.pickle", "wb") as f:
#     pickle.dump([train_x, train_y, test_x, test_y], f)

# placeholder. Shape = height x width. 
x = tf.placeholder("float", shape=[None, len(train_x[0])])
y = tf.placeholder("float")

# call function train neural network
train_neural_network(x, y, test_x, test_y)

    
