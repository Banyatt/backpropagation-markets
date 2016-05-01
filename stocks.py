import math
from numpy import zeros
from numpy import random

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
    self.input_weights = random.rand(self.inputs, self.hidden)

    # Initialize weight matrix between hidden units and outputs
    self.output_weights = random.rand(self.hidden, self.outputs)

    # Learning rate alpha
    self.alpha = 0.5

    # Iterations to train network for
    self.iterations = 1000

  def update(self, inputs):
    # Compute activation for all inputs except bias
    for input in range(self.inputs - 1):
      self.input_activations[input] = inputs[input]

    # Compute activation for all hidden units
    for hidden in range(self.hidden):
      sum = 0.0
      for input in range(self.inputs):
          sum += self.input_activations[input] * self.input_weights[input][hidden]
      self.hidden_activations[hidden] = sigmoid(sum)

    # Compute activation for all output units
    for output in range(self.outputs):
      sum = 0.0
      for hidden in range(self.hidden):
        sum += self.hidden_activations[hidden] * self.output_weights[hidden][output]
      self.outputs_activations[output] = sigmoid(sum)

    return self.outputs_activations

  def backpropagate(self, targets):
    # Compute error at output units
    output_deltas = [0.0] * self.outputs
    for output in range(self.outputs):
      error = targets[output] - self.outputs_activations[output]
      output_deltas[output] = deltaSigmoid(self.outputs_activations[output]) * error

    # Compute error at hidden units
    hidden_deltas = [0.0] * self.hidden
    for hidden in range(self.hidden):
      error = 0.0
      for output in range(self.outputs):
        error += output_deltas[output] * self.output_weights[hidden][output]
      hidden_deltas[hidden] = deltaSigmoid(self.hidden_activations[hidden]) * error

    # Update output unit weights
    for hidden in range(self.hidden):
      for output in range(self.outputs):
        update = output_deltas[output] * self.hidden_activations[hidden]
        self.output_weights[hidden][output] = self.output_weights[hidden][output] + self.alpha * update

    # Update input unit weights
    for input in range(self.inputs):
      for hidden in range(self.hidden):
        update = hidden_deltas[hidden] * self.input_activations[input]
        self.input_weights[input][hidden] = self.input_weights[input][hidden] + self.alpha * update

    # Compute total error
    error = 0.0
    for target in range(len(targets)):
      error += 0.5 * (targets[target] - self.outputs_activations[target]) ** 2
    return error

  def test(self, patterns):
    for pattern in patterns:
      print(pattern[0], '->', self.update(pattern[0]))

  def train(self, patterns):
    for iteration in range(self.iterations):
      error = 0.0
      for pattern in patterns:
        inputs = pattern[0]
        targets = pattern[1]
        self.update(inputs)
        error += self.backpropagate(targets)
      if iteration % 100 == 0:
        print('error %-.5f' % error)

def run():
  # Teach network XOR function
  patterns = [
    [[0,0], [0]],
    [[0,1], [1]],
    [[1,0], [1]],
    [[1,1], [0]]
  ]

  # Create network architecture consisting of 2 input nodes, 2 hidden units, and 1 output unit
  n = network(2, 2, 1)

  # Train the network
  n.train(patterns)

  # Test the network
  n.test(patterns)

run()