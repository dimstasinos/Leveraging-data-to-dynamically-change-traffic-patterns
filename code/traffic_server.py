

import torch
import pickle
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import torch.nn as nn
import warnings
import neural_algorithms as nn_algo
warnings.simplefilter(action='ignore', category=FutureWarning)


class Server:
    def __init__(self, n_clients, option):
        self.n_clients = n_clients

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        #choose the model based on the first recieved message from the single client process
        if option == "1":
            self.global_model = nn_algo.DeepNN().to(self.device)
            self.model_name = 'deepNN'
        else:
            self.global_model = nn_algo.GRU().to(self.device)
            self.model_name = 'gru'

        self.global_weights = self.global_model.state_dict()

        self.producer = KafkaProducer(bootstrap_servers='localhost:9092')

        self.consumer = KafkaConsumer(bootstrap_servers='localhost:9092', group_id='server',
                                     enable_auto_commit=False, auto_offset_reset='earliest')

    # read the weights on the local topic,read from the last position
    def recieve_weights(self, topic='local-weights'):
        local_weights = []
        for partition in range(self.n_clients):
            
            assigned_partition = [TopicPartition(topic, partition)]
            self.consumer.assign(assigned_partition)

            # you read all the messages that are directed for you and you store them in a list
            for message in self.consumer:

                data = pickle.loads(message.value)
                partition_id = data['client_info']
                local_weight = data['weights']
                local_weights.append(local_weight)
                print(f"got the message from {partition_id} on {partition}")
                self.consumer.commit() #commmit in each partition an offset of the last read message
                break

        return local_weights

    def aggregate_weights(self, local_weights_list):
        # initialize with zero's
        aggregated_weights = {key: torch.zeros_like(
            value) for key, value in self.global_weights.items()}

        # simple federated averaging of the weights
        for local_weights in local_weights_list:
            for key in self.global_weights.keys():
                aggregated_weights[key] += local_weights[key]

        for key in aggregated_weights.keys():
            aggregated_weights[key] /= self.n_clients

        # save the wights and update the model
        self.global_weights = aggregated_weights
        self.global_model.load_state_dict(self.global_weights)

        torch.save(self.global_model.state_dict(),
                   f"NN_models/global_model_{self.model_name}.pth")
        print("Server: Aggregated and saved model weights.")

    # send the weights back to each traffic light

    def send_weights_back(self, topic='global-weights'):

        global_weights_serialized = pickle.dumps(self.global_weights)

        for partition in range(self.n_clients):
            self.producer.send(
                topic, value=global_weights_serialized, partition=partition)
        self.producer.flush()
        print("Weights Send")

    def initialize_server(self):
        # initialzie the communication form the server side(start the "ping pong")
        rounds = 10
        for round in range(rounds):
            print(f"round: {round+1}")
            local_weights_list = self.recieve_weights()
            self.aggregate_weights(local_weights_list)

            # in the last round we just need the global model dont need to update the local ones
            if round < (rounds-1):
                self.send_weights_back()
            else:
                self.consumer.close()

                self.producer.flush()
                self.producer.close()



def read_first_message():
    nn_consumer = KafkaConsumer("handshake", bootstrap_servers='localhost:9092',
                                enable_auto_commit=True,group_id='handshake_group',auto_offset_reset='earliest')
    option = None

    message = next(nn_consumer)
    option = message.value.decode('utf-8')

    nn_consumer.close()

    return option


if __name__ == "__main__":

    print("----Waiting for the user to select a nn model----")
    n_clients = 4
    option = read_first_message()

    print("--> NN option received")
    server = Server(n_clients, option)
    server.initialize_server()
    print("Server: Finished")
