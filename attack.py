# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 

@authors: Ammar Kamal Abasi , Neveen Mohammad Hijazi, Moayad Aloqaily, Mohsen Guizani
Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI), UAE
E-mails: {ammar.abasi; neveen.hijazi; moayad.aloqaily; mohsen.guizani}@mbzuai.ac.ae
"""


#With FGSM
import hashlib
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from time import time
import random
import pandas as pd
import os
os.environ['TF_DISABLE_THREADING'] = '1'
loss_object = tf.keras.losses.MeanSquaredError()

def fgsm(model, input_instance, label, epsilon =0.5):
    tensor_input_instance = tf.convert_to_tensor(input_instance, dtype=tf.float32)
    adv_x = input_instance
    for idx in range(0,len(label) ):#     
      with tf.GradientTape() as tape:
          tmp_label = label[idx]
          tape.watch(tensor_input_instance)
          prediction = model(tensor_input_instance)
          loss = loss_object(tmp_label, prediction)
          gradient = tape.gradient(loss, tensor_input_instance)
          signed_grad = tf.sign(gradient)
          adv_x = adv_x + eps * signed_grad
    return adv_x




class Block:
    def __init__(self, index, timestamp, data, previous_hash=''):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """
        Calculates the hash of the block using SHA-256.
        """
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

    def create_genesis_block(self):
        """
        Creates the first block in the blockchain.
        """
        return Block(0, time(), {"accuracy": 0, "weights": []}, "0")

    def get_latest_block(self):
        """
        Returns the latest block in the blockchain.
        """
        return self.chain[-1]

    def add_block(self, data):
        """
        Adds a new block to the blockchain.
        """
        index = len(self.chain)
        previous_hash = self.get_latest_block().hash
        new_block = Block(index, time(), data, previous_hash)
        self.chain.append(new_block)

    def add_transaction(self, transaction):
        """
        Adds a new transaction to the pending transactions list.
        """
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self):
        """
        Mines all the pending transactions and adds them to the blockchain as a new block.
        """
        data = self.pending_transactions.copy()
        self.pending_transactions = []
        self.add_block(data)

    def get_block_by_index(self, index):
        """
        Returns the block with the given index.
        """
        return self.chain[index]

    def get_block_by_hash(self, hash):
        """
        Returns the block with the given hash.
        """
        for block in self.chain:
            if block.hash == hash:
                return block

        return None

class CNNModel():
    def __init__(self):
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Split the training data and labels into 10 equal parts
        x_train = np.split(x_train, 1)
        y_train= np.split(y_train, 1)
        
        # Split the testing data and labels into 10 equal parts
        x_test= np.split(x_test, 1)
        y_test = np.split(y_test, 1)
        
        # Use only the first part of the data
        x_train, y_train = x_train[0], y_train[0]
        x_test, y_test = x_test[0], y_test[0]

        num_samples = x_train.shape[0]
        print("Number of samples in x_train of the dataset:", num_samples)
        # Normalize the pixel values to be between 0 and 1
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Convert the labels to one-hot encodings
        y_train = np.eye(10)[y_train.reshape(-1)]
        y_test = np.eye(10)[y_test.reshape(-1)]



        # Store the dataset
        self.dataset = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
        
        # Initialize the weights
        self.weights = [np.random.randn(3, 3, 3, 64), np.random.randn(64), np.random.randn(64, 10), np.random.randn(10)]

        # Store the dataset
        self.dataset = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

        # Initialize the weights
        #self.weights = [np.random.randn(3, 3, 3, 64), np.random.randn(64), np.random.randn(64, 10), np.random.randn(10)]
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])
    def set_weights(self, weights):
        # Set the model weights
        self.model.set_weights(weights)

    def get_weights(self):
        # Get the model weights
        return self.model.get_weights()
    def select_device_data(self,N,device_num,num_rounds,round_num):
        x_train, y_train = self.dataset["x_train"], self.dataset["y_train"]
        x_test, y_test = self.dataset["x_test"], self.dataset["y_test"]
        
        sample_size = 5000  # Set the size of the sample you want
        random_indices = np.random.choice(x_train.shape[0], sample_size, replace=False)
        x_train = x_train[random_indices]
        num_samples = x_train.shape[0]
        print(f"Number of samples in x_train for device {device_num}: {num_samples}")
        #y_train = y_train[start_train:end_train]
        y_train =  y_train[random_indices]
        #start_test = device_num*N_test
        #end_test = (device_num+1)*N_test 
        sample_size2 = 1000 
        random_indices2 = np.random.choice(x_test.shape[0], sample_size2, replace=False)
        #x_test = x_test[start_test:end_test]
        #y_test = y_test[start_test:end_test]
        x_test = x_test[random_indices2]
        y_test = y_test[random_indices2]
        return x_train, y_train, x_test, y_test
    
    def train(self, N, device_num,num_rounds,round_num):
        # TODO: train the model on the CIFAR-10 dataset
        # update the weights of the model
        # return the accuracy and weights
        x_train, y_train, x_test, y_test = self.select_device_data(N, device_num,num_rounds,round_num)
       
        # Define the model architecture
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#mean_squared_error
        # Train the model
        model.fit(x_train, y_train, verbose=0,epochs=10, validation_data=(x_test, y_test))

        # Get the accuracy and weights
        accuracy = model.evaluate(x_test, y_test)[1]
        weights = model.get_weights()

        return x_train, y_train,x_test, y_test,model, accuracy, weights
    
    def train2(self,model,In_test_adv,y_test):

        
        # Get the accuracy and weights
        accuracy = model.evaluate(In_test_adv, y_test)[1]
        weights = model.get_weights()

        return  accuracy, weights

# create a new blockchain
blockchain = Blockchain()

# Create the initial global model
global_model = CNNModel()
num_rounds = 10
N = 20
# Perform federated learning for each round
for round_num in range(num_rounds):
    print("Round", round_num+1)
    # create N instances of the CNN model

    #print(global_model.get_weights())
    models = [global_model for _ in range(N)]
    
    # train each model on its own local dataset
    random_client=random.sample(range(0, 20), random.randint(10, 19))
    for i in range(N):
        x_train, y_train, x_test, y_test, model, accuracy, weights = models[i].train(N, i,num_rounds,round_num)
        #print("i=",i)
        
        if i in (random_client) and round_num>=1: #odd numbers
          #print("i2=",i)
          eps = 2.0 * 16.0 / 255.0
          In_test_adv = fgsm(model, x_test,y_test,eps)
          accuracy, weights = models[i].train2(model,In_test_adv,y_test)
        # create a transaction with the accuracy and weights
        transaction = {"device": i, "accuracy": accuracy, "weights": [w.tolist() for w in weights]}
    
        # add the transaction to the pending transactions list
        blockchain.add_transaction(transaction) 
    
    # mine the pending transactions and add them to the blockchain as a new block
    blockchain.mine_pending_transactions()
    
    # get the latest block from the blockchain
    latest_block = blockchain.get_latest_block()
    
    # extract the accuracy and weights from each device's transaction in the latest block
    accuracies = [0] * N
    weights = [None] * N
    for transaction in latest_block.data:
        device = transaction["device"]
        accuracy = transaction["accuracy"]
        w = transaction["weights"]
        accuracies[device] = accuracy
        weights[device] = [np.array(w[i]) for i in range(len(w))]
    
    # Federated Averaging
    global_weights = np.mean(weights, axis=0)
    
    # update each model with the aggregated weights
    #for i in range(N):
        #models[i].weights = global_weights.tolist()
    global_model.set_weights(global_weights)
    
    # print the final accuracy and weights
    print("Final accuracy:", np.mean(accuracies))
    output = pd.DataFrame({"attack":[np.mean(accuracies)],"k":[N]})
    output.to_csv(os.path.join("output", "attack.csv"), mode='a', index=False,header=False) 
    #print("Final weights:", global_weights.tolist())