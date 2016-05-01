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