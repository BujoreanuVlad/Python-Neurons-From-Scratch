inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]

#Initialize output list
outputs = []

#For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    
    #Initialize output variable of the current neuron
    neuron_output = 0

    #For each input and weight of the current neuron
    for neuron_input, neuron_weight in zip(inputs, neuron_weights):
        neuron_output += neuron_input * neuron_weight
    neuron_output += neuron_bias

    #Append the result to the output list
    outputs.append(neuron_output)


print(outputs)