import math
import random

def sigmoid(x):
  return math.tanh(x)

def deltaSigmoid(y):
  return 1.0 - y**2