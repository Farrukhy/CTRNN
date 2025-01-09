import numpy as np
import matplotlib.pyplot as plt
from CTRNN import CTRNN

# Simple Hello World Script using CTRNN

# Params for a simple network
run_duration = 250
net_size = 2
step_size = 0.01

# Set up the network
network = CTRNN(size=net_size, step_size=step_size)
network.taus = [1.0, 1.0]
network.biases = [-2.75, -1.75]
network.weights[0, 0] = 4.5
network.weights[0, 1] = 1.0
network.weights[1, 0] = -1.0
network.weights[1, 1] = 4.5

# Initialize network
network.randomize_outputs(0.1, 0.2)

# Simulate network
outputs = []
for _ in range(int(run_duration / step_size)):
    network.euler_step([0] * net_size)  # Zero external inputs
    outputs.append([network.outputs[i] for i in range(net_size)])
outputs = np.asarray(outputs)

# Plotting the oscillator output
plt.plot(np.arange(0, run_duration, step_size), outputs[:, 0])
plt.plot(np.arange(0, run_duration, step_size), outputs[:, 1])
plt.xlabel('Time')
plt.ylabel('Neuron Outputs')
plt.title('CTRNN Oscillator Output')
plt.show()
