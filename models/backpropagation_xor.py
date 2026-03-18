# -*- coding: utf-8 -*-
"""
Backpropagation on XOR
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("XOR Training Data:")

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.7):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.losses = []
        self.accuracies = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        self.dz2 = output - y
        self.dW2 = (1/m) * np.dot(self.a1.T, self.dz2)
        self.db2 = (1/m) * np.sum(self.dz2, axis=0, keepdims=True)
        
        self.da1 = np.dot(self.dz2, self.W2.T)
        self.dz1 = self.da1 * self.sigmoid_derivative(self.a1)
        self.dW1 = (1/m) * np.dot(X.T, self.dz1)
        self.db1 = (1/m) * np.sum(self.dz1, axis=0, keepdims=True)
    
    def update_weights(self):
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
    
    def train(self, X, y, epochs=5000, verbose=True):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            self.losses.append(loss)
            
            predictions = (output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            self.accuracies.append(accuracy)
            
            self.backward(X, y, output)
            self.update_weights()
            
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | Accuracy: {accuracy:.2%}")
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

print("Training Neural Network...")
nn = NeuralNetwork(learning_rate=0.7)
nn.train(X, y, epochs=5000)

predictions = nn.predict(X)
print("
Final Results:")
for i in range(len(X)):
    correct = "✓" if predictions[i][0] == y[i][0] else "✗"
    print(f"{X[i][0]} XOR {X[i][1]} = {predictions[i][0]} (true: {y[i][0]}) {correct}")

accuracy = np.mean(predictions == y)
print(f"
Final Accuracy: {accuracy:.2%}")
