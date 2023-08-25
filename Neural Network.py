#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.special


# In[81]:


class neuralNetwork:
    def __init__(self,inputnodes,outputnodes,hiddennodes,learningrate):
        self.inode = inputnodes
        self.onode = outputnodes
        self.hnode = hiddennodes
        self.lr = learningrate
        self.wih = np.random.normal(0.0,pow(self.hnode,-0.5),(self.hnode,self.inode))
        self.who = np.random.normal(0.0,pow(self.onode,-0.5),(self.onode,self.hnode))
        self.activation_function = lambda x : scipy.special.expit(x)
        
    def train(self,inputs_list,targets_list):
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_error = np.dot(self.who.T,output_errors)
        
        self.who += self.lr*np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr*np.dot((hidden_error*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
        
    def query(self,inputs_list):
        inputs = np.array(inputs_list,ndmin=2).T
        
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    


# In[82]:


inputnodes = 784
outputnodes = 10
hiddennodes = 100

learning_rate = 0.3

n = neuralNetwork(inputnodes,outputnodes,hiddennodes,learning_rate)


# In[83]:


training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[52]:


all_values = data_list[1].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')


# In[89]:


# Training the model
for record in training_data_list :
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:])/(255.0*0.99) + 0.01)
    targets = np.zeros(outputnodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass


# In[85]:


test_data_file = open("mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[74]:


all_values = test_data_list[0].split(',')
print(all_values[0])


# In[75]:


n.query((np.asfarray(all_values[1:])/255.0*0.99) + 0.01)


# In[90]:


# Testing the model
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)


# In[91]:


scorecard_array = np.asarray(scorecard)
print ("performance = {}%".format( (scorecard_array.sum() /
scorecard_array.size)*100))


# In[ ]:




