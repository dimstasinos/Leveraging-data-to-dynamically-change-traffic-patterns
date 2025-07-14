# FEDERATED TRAFFIC-PATTERN LEARNING SYSTEM 

This project aims to make a federated learning system that takes all the densities of all the roads 
connected to a local traffic light,individually train the local neural network of the light, do this 
exact process for all the traffic lights and then  establish a communication for a certain amount of rounds
with a server in order to average all the weights and redistribute them back.
When the whole process is finished we have a global model that can be applied to any light so it can control
its timings. Thats possible because after the rounds of communication and re-training ,the model has learned the global
characteristics from every local light, so the prediction for the timings is made with knoweldge of the whole traffic system.

## Architecture

The project architecture consists of:

- **4 traffic light clients** running in parallel threads
- **Kafka server** managing communication via 3 topics
- **Federated averaging** logic on the server side
- **Neural networks** (DeepNN and GRU-based) trained per intersection and aggregated centrally

## Training and Communication Flow

1. Each traffic light loads local `.pkl` datasets with traffic density and timing values.
2. Training is performed locally with loss evaluation after each epoch.
3. After each training round, local weights are sent to the Kafka server.
4. Server aggregates weights and returns the updated global model.
5. Process repeats for a fixed number of rounds (e.g., 10).
6. Final global model is used in the simulation for real-time decision making.
