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