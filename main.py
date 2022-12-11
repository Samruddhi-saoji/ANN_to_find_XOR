#better accuracy

#training an ANN to find XOR of 2 numbers
import numpy as np
from scipy.special import expit


#the activation function############################
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):  #derivation of activation function
    return expit(x)*(1-expit(x))
#used in back propagation



##### the neural network ################################
layers = 3
#number of nuerons in each layer
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1


############ initialise weights and biases ##########
#hidden layer
hidden_weights = np.random.rand(hiddenLayerNeurons,inputLayerNeurons) -0.5
hidden_bias = np.random.rand(hiddenLayerNeurons,1) -0.5 
#output layer
output_weights = np.random.rand(outputLayerNeurons,hiddenLayerNeurons) -0.5  
output_bias = np.random.rand(outputLayerNeurons,1) -0.5
#random values have been assigned to the weights and biases of each layer


############ training the ANN on the train dataset ###########
def train(X,Y, epochs, lr) :
    #declaring global variables
    global hidden_weights, hidden_bias, output_weights, output_bias

    #input and expected outputs
    input_activations = X
    expected_output = Y

    #number of training observations
    m = X.size

    #training episodes
    for _ in range(epochs) :
        #### forward propagation ##########################
        #hidden_layer
        hidden_z = np.dot(hidden_weights, input_activations) + hidden_bias
        hidden_activations = sigmoid(hidden_z) #expit(x) means sigmoid(x)

        #output layer
        output_z = np.dot(output_weights , hidden_activations) + output_bias
        predicted_output = sigmoid(output_z) #output layer activation


        #### back propagation ##############################
        #calculate the gradient
        #output layer
        error = predicted_output - expected_output  #dz_output
        dW_output = np.dot(error, hidden_activations.T)
        dB_output = np.sum(error, 1)

        #hidden_layer
        hidden_layer_error = np.dot(output_weights.T , error)*sigmoid_derivative(hidden_z) 
        dW_hidden = np.dot(hidden_layer_error ,input_activations.T)
        dB_hidden = np.sum(hidden_layer_error, 1)


        ######## updating weights and biases a/c to gradient #######
        # formula: w = w - lr*dW
        #hidden layer
        hidden_weights = hidden_weights - lr*(dW_hidden)
        hidden_bias = hidden_bias - (lr)*np.reshape(dB_hidden , (hiddenLayerNeurons, 1))

        #output layer
        output_weights = output_weights - (lr)*dW_output
        output_bias = output_bias - (lr)*np.reshape(dB_output, (outputLayerNeurons,1))
    #training completed

    #print the final weights and biases
    print("\nFinal hidden weights: ",*hidden_weights)
    print("Final hidden bias: ",*hidden_bias)
    print("Final output weights: ",*output_weights)
    print("Final output bias: ", *output_bias)

    #print the final predicted output
    print("\n\nOutput from neural network after 1000 epochs: ", *predicted_output)

    #return the final values of the weights and biases
    return hidden_weights, hidden_bias , output_weights , output_bias



############# driver code ##################################
#training data
X = np.array([[0,0,1,1], [0,1,0,1]])
Y = np.array([0,1,1,0])

train(X, Y, 10000, 0.1)

