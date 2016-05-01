import math
import random
from numpy import zeros

def sigmoid(x):
  return math.tanh(x)

def deltaSigmoid(y):
  return 1.0 - y**2

class network:
  def __init__(self, inputs, hidden, outputs):
    # Initialize inputs as well as bias
    self.inputs = inputs + 1

    # Initialize hidden units
    self.hidden = hidden

    # Initialize output units
    self.outputs = outputs

    # Initialize activation for input nodes
    self.input_activations = [1.0] * self.inputs

    # Initialize activation for hidden units
    self.hidden_activations = [1.0] * self.hidden

    # Initialize activation for output units
    self.outputs_activations = [1.0] * self.outputs

    # Initialize weight matrix between inputs and hidden units
    self.input_weights = zeros(self.inputs, self.hidden)

    # Initialize weight matrix between hidden units and outputs
    self.output_weights = zeros(self.hidden, self.outputs)

    # Learning rate alpha
    self.alpha = 0.5

  def update(self, inputs):
    # Compute activation for all inputs except bias
    for neuron in range(self.inputs - 1):
      self.input_activations[neurons]

    # Compute activation for all hidden units
    for unit in range(self.hidden):
      sum = 0.0
      for input in range(self.inputs):
        sum += self.input_activations[input] * self.input_weights[input][unit] 
      self.hidden_activations[input] = sigmoid(sum)

    # Compute activation for all output units
    for output in range(self.outputs):
      sum = 0.0
      for hidden in range(self.hidden):
        sum += self.hidden_activations[hidden] * self.output_weights[hidden][output]
      self.outputs_activations[output] = sigmoid(sum)

  def backpropagate(self, targets):
    # Compute output error
    delta_output = [0.0] * self.outputs
    for output in range(self.outputs):
      error = targets[output] - self.outputs_activations[output]
      delta_output[output] = deltaSigmoid(self.outputs_activations[output]) * error

    # Compute hidden unit error
    delta_hidden = [0.0] * self.hidden
    for hidden in range(self.hidden):
      error = 0.0
      for output in range(self.outputs):
        error += delta_output[output] * self.output_weights[hidden][output]
      delta_hidden[hidden] = deltaSigmoid(self.hidden_activations[hidden]) * error

    # Update the input weights
    for input in range(self.inputs):
      for hidden in range(self.hidden):
        update = delta_hidden[hidden] * self.input_activations[input]
        self.input_weights[input][hidden] + self.alpha * update

    # Compute total error
    error = 0.0
    for target in range(len(targets)):
      error += 0.5 * (targets[target] - self.outputs_activations[target]) ** 2
    return error

  def test(self, patterns):
    for pattern in patterns:
      print p[0], '->', self.update(p[0])

  def weights(self):
    print 'Input weights'
    for input in range(self.inputs):
      print self.input_weights[input]
    print 'Output Weights'
    for hidden in range(self.hidden):
      print self.output_weights[hidden]

