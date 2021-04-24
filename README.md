# siarne
A library for fast simulation of artificial neurons.

## Networks
A network is a ring of neurons where each neuron has a connection to itself and 0 or more other neurons neighbouring it.  
At each `tick` the input of each neuron (32 bit signed integer) is compared to it's treshold, if it's greater then the neuron adds effects (8 bit signed integers) 
to the inputs of neurons at the ends of its connections.  

The parameters of a network are the effects, tresholds and the index of the neurons from which input and output is extracted.
There are utilities for tweaking these parameters (training), currently only an evolution strategy is implemented.

## Optimization
The main evaluation code is automatically vectorized and should therefore run blazing fast on many platforms.  
To give an impression, the `network_tick` benchmark on an i7-4790 runs at about 370 microseconds per iteration for 4096 neurons with 1024 connections per neuron. That's 11 billion connections per second.  
If the benchmark is run with `target-cpu=native` the time per tick goes down to about 200 microseconds, or 21 billion connections per second.  

Currently the evaluation code is completely single threaded, future work includes looking at multithreading oppurtunities within a single network.  

For the training code performance is considered less important and e.g quality of random numbers takes priority.
