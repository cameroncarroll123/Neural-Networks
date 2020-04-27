# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:44:18 2020

@author: Cameron

Activation Function: Sigmoid
Loss Function: Sum-of-Squares Error
Biases = 0
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def sigmoidDer(x):
  return x * (1.0-x)

class NeuralNetwork:
  def __init__(self,x,y):
    self.input = x
    self.weights1 = np.random.rand(self.input.shape[1],y.size)
    self.weights2 = np.random.rand(y.size,1)
    self.y = y
    self.output = np.zeros(y.shape)
    self.errorHistory = []
    self.itsList = []
    
  def feedforward(self):
    self.layer1 = sigmoid(self.input @ self.weights1)
    self.output = sigmoid(self.layer1 @ self.weights2)
    
  def backprop(self):
    self.error = self.y - self.output
    loss = 2*(self.error)*sigmoidDer(self.output)
    d_weights2 = self.layer1.T @ loss
    d_weights1 = self.input.T @ ((loss @ self.weights2.T) * sigmoidDer(self.layer1))
    
    self.weights1 += d_weights1
    self.weights2 += d_weights2
    
  def train(self, its=25000):
    for it in range(its):
      self.feedforward()
      self.backprop()
      self.errorHistory.append(np.average(np.abs(self.error)))
      self.itsList.append(it)
  
  def predict(self, newInput):
    predictLayer1 = sigmoid(newInput @ self.weights1)
    prediction = sigmoid(predictLayer1 @ self.weights2)
    return prediction
    
    
x = np.array([[0, 1, 0],
              [0, 1, 1],
              [0, 0, 0],
              [1, 0, 0],
              [1, 1, 1],
              [1, 0, 1]])

y = np.array([[0],
               [0],
               [0],
               [1],
               [1],
               [1]])

nn = NeuralNetwork(x,y)
nn.train()

ex1 = np.array([[1,1,0]])
ex2 = np.array([[0,0,1]])

print(nn.predict(ex1), ' - Correct: ', ex1[0][0])
print(nn.predict(ex2), ' - Correct: ', ex2[0][0])

print(nn.weights1)
print(nn.weights2)
# print(nn.output)
plt.figure()
plt.plot(nn.itsList, nn.errorHistory)
