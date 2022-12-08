#training an ANN to find XOR of 2 numbers
import numpy as np 

#the activation function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):  #derivation of activation function
    return x * (1 - x)
#used in back propagation


#Input dataset
inputs = np.array([[0,0],[0,1],[1,0],[1,1]]) #aka the input layer activation
expected_output = np.array([[0],[1],[1],[0]])

#hyper parameters
epochs = 10000
lr = 0.1  #learning rate


########################## creating the nueral network #############################3
layers = 3
#number of nuerons in each layer
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

#the weights and biases
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))
#random values have been assigned to the weights and biases of each layer

#print the initial weights and biases
print("Initial hidden weights: ", *hidden_weights)
print("Initial hidden bias: ",*hidden_bias)
print("Initial output weights: ",*output_weights)
print("Initial output bias: ", *output_bias)


########################################### Training algorithm #########################
#for each training episode: 
for _ in range(epochs):
    ########### Forward Propagation ##############
    hidden_layer_activation = sigmoid (np.dot(inputs,hidden_weights) + hidden_bias)

    #the output layer activation
    predicted_output = sigmoid(np.dot(hidden_layer_activation,output_weights) + output_bias)

    ##################################################### study from here
    ########### backward Propagation #############
    #error of output layer
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    #error of hidden layer
    error_hidden_layer = np.dot(d_predicted_output, output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

    #Updating Weights and Biases ###############
    output_weights = output_weights + (lr)*( np.dot(hidden_layer_activation.T, d_predicted_output) )
    output_bias = output_bias +  (lr)*(np.sum(d_predicted_output,axis=0,keepdims=True))  #computes the per-column sums 

    hidden_weights = hidden_weights + (lr)*(np.dot(inputs.T , d_hidden_layer))
    hidden_bias = hidden_bias + (lr)*(np.sum(d_hidden_layer,axis=0,keepdims=True)) #formula of derivative (more precisely gradient) of the loss function with respect to the bias
#training completed

#print the final weights and biases
print("\nFinal hidden weights: ",*hidden_weights)
print("Final hidden bias: ",*hidden_bias)
print("Final output weights: ",*output_weights)
print("Final output bias: ", *output_bias)

#print the final predicted output
print("\n\nOutput from neural network after 10,000 epochs: ", *predicted_output)