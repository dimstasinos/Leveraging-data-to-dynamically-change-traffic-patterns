# FEDERATED TRAFFIC-PATTERN LEARNING SYSTEM 

This project aims to make a federated learning system that takes all the densities of all the roads 
connected to a local traffic light,individually train the local neural network of the light, do this 
exact process for all the traffic lights and then  establish a communication for a certain amount of rounds
with a server in order to average all the weights and redistribute them back.
When the whole process is finished we have a global model that can be applied to any light so it can control
its timings. Thats possible because after the rounds of communication and re-training ,the model has learned the global
characteristics from every local light, so the prediction for the timings is made with knoweldge of the whole traffic system.
