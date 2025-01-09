# imports
import numpy as np
import matplotlib.pyplot as plt
# importing the CTRNN class
from CTRNN import CTRNN

# params
run_duration = 250
net_size = 2
step_size = 0.01

# set up network
network = CTRNN(size=net_size,step_size=step_size)
network.taus = [1.,1.]
network.biases = [-2.75,-1.75]
network.weights[0,0] = 4.5
network.weights[0,1] = 1
network.weights[1,0] = -1
network.weights[1,1] = 4.5

# network.weights[0,1] = 2.0  # Increase influence of neuron 1 on neuron 2
# network.weights[1,0] = -2.0  # Make inhibition from neuron 2 to neuron 1 stronger

# initialize network
network.randomize_outputs(0.1,0.2)


# simulate network
outputs = []
for _ in range(int(run_duration/step_size)):
    network.euler_step([0]*net_size) # zero external_inputs
    outputs.append([network.outputs[i] for i in range(net_size)])
outputs = np.asarray(outputs)

# plot oscillator output
plt.plot(np.arange(0,run_duration,step_size),outputs[:,0])
plt.plot(np.arange(0,run_duration,step_size),outputs[:,1])
plt.xlabel('Time')
plt.ylabel('Neuron outputs')
plt.show()

external_inputs = [np.sin(0.1 * t) for t in range(int(run_duration / step_size))]
for i, ext_input in enumerate(external_inputs):
    network.euler_step([ext_input, 0])  # Sinusoidal input to neuron 0
