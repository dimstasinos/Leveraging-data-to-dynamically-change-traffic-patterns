import matplotlib.pyplot as plt
import matplotlib
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import neural_algorithms as nn_algo
import pickle
import shutil
import threading
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

matplotlib.use('Agg')


# Load the global scalers
with open('scalers/densities_scaler.pkl', 'rb') as f:
    densities_scaler = pickle.load(f)

with open('scalers/timings_scaler.pkl', 'rb') as f:
    timings_scaler = pickle.load(f)


class TrafficLight:
    def __init__(self, light_id, training_file, option):
        self.light_id = light_id
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Select the nn model
        if option == "1":
            self.neural = nn_algo.DeepNN().to(self.device)
        else:
            self.neural = nn_algo.GRU().to(self.device)
        
        self.optimizer = optim.Adam(self.neural.parameters(), lr=0.001)
        self.criterion = nn.MSELoss().to(self.device)

        #each thread's consumer and producer
        self.kafka_consumer = KafkaConsumer(
            bootstrap_servers='localhost:9092',
            group_id='client',
            enable_auto_commit=False,
            auto_offset_reset='earliest'
        )
        self.kafka_producer = KafkaProducer(bootstrap_servers='localhost:9092')

        # Load and preprocess the data for each traffic light
        self.data = []
        with open(training_file, 'rb') as f:
            while True:
                try:
                    self.data.append(pickle.load(f))
                except EOFError:
                    break

        self.data_inputs = []
        self.data_targets = []

        for row in self.data:
            densities, timings = row
            self.data_inputs.append(densities)
            self.data_targets.append(timings)

        # Normalize the data using the global scalers and split into train/validation sets
        self.normalize_and_split_data()

        # Prepare the data for training
        train_densities = torch.tensor(
            self.train_densities, dtype=torch.float32).to(self.device)
        train_timings = torch.tensor(
            self.train_timings, dtype=torch.float32).to(self.device)

        train_densities = TensorDataset(train_densities, train_timings)
        self.train_dataloader = DataLoader(
            train_densities, batch_size=32, shuffle=True)

        # Prepare the validation data
        val_densities = torch.tensor(
            self.val_densities, dtype=torch.float32).to(self.device)
        val_timings = torch.tensor(
            self.val_timings, dtype=torch.float32).to(self.device)

        val_dataset = TensorDataset(val_densities, val_timings)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=32, shuffle=False)

    def normalize_and_split_data(self):
        # Use global scalers to transform the data
        self.densities_normalized = densities_scaler.transform(
            self.data_inputs)
        self.timings_normalized = timings_scaler.transform(self.data_targets)

        # Split the data into training and validation sets
        self.train_densities, self.val_densities, self.train_timings, self.val_timings = train_test_split(
            self.densities_normalized, self.timings_normalized, test_size=0.2, random_state=42
        )

    # Train the nn model
    def training(self, epochs):
        train_losses = []
        val_losses = []
        for _ in range(epochs):
            self.neural.train()
            total_loss = 0
            for densities_batch, timings_batch in self.train_dataloader:
                densities_batch = densities_batch.to(self.device)
                timings_batch = timings_batch.to(self.device)

                #simple training of the local model
                self.optimizer.zero_grad()#resetting the gradients
                outputs = self.neural(densities_batch)
                loss = self.criterion(outputs, timings_batch)#the error between the target values and the predictions
                loss.backward() #backpropagation of loss
                self.optimizer.step()
                total_loss += loss.item()

            #compute and append the training and evaluation losses to compare them and plot them later
            avg_loss = total_loss / len(self.train_dataloader)
            train_losses.append(avg_loss)

            # Validation phase
            self.neural.eval()
            val_loss = 0
            with torch.no_grad():
                for densities_batch, timings_batch in self.val_dataloader:
                    densities_batch = densities_batch.to(self.device)
                    timings_batch = timings_batch.to(self.device)
                    outputs = self.neural(densities_batch)
                    loss = self.criterion(outputs, timings_batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_dataloader)
            val_losses.append(avg_val_loss)

        return train_losses, val_losses  # Return the losses for plotting

    # Get the current weights
    def get_weights(self):
        return self.neural.state_dict()

    # Set the new weights to the model
    def set_weights(self, new_weights):
        self.neural.load_state_dict(new_weights)

    # Send the weights to the server
    def send_weights(self, partition, topic='local-weights'):
        
        weights = self.get_weights()

        metadata = {'client_info': f'partition:{partition}',
                    'weights': weights}
        formatted_message = pickle.dumps(metadata) #create metadata in order to know which client/partition sent the message and the actual data
        self.kafka_producer.send(topic, value=formatted_message,
                            partition=partition)
        self.kafka_producer.flush() #flush the buffer
        print(f"Weights sent successfully to partition {partition}")

    # Receive the new weights from the server
    def receive_global_weights(self, partition, topic='global-weights'):

        assigned_partition = [TopicPartition(topic, partition)] #assign the consumer of each light on the correct partition
        self.kafka_consumer.assign(assigned_partition)

        for message in self.kafka_consumer:
            global_weights = pickle.loads(message.value)
            self.set_weights(global_weights)
            print(f"Client {self.light_id}: Updated with global weights.")
            self.kafka_consumer.commit() #commit an offset of what we read
            break

    def initialize_nn(self, partition):
        # Initialize the communication from the client side
        rounds = 10
        starting_epochs = 100

        if partition==0:
            print(f"Running the program for {rounds} rounds and {starting_epochs} epochs")

        for round in range(rounds):
            
            if partition==0:
                print(f"Round: {round + 1}")
            
            train_losses, val_losses = self.training(
                starting_epochs-10*round)  # Get losses from training
            self.save_loss_plot(train_losses, val_losses,
                                round + 1)  # Save the plot

            print(f"Send weights {self.light_id}")
            self.send_weights(partition=partition)

            # Last round doesn't need to receive any messages, just need the global model
            if round < (rounds - 1):
                self.receive_global_weights(partition=partition)
            else:
                self.kafka_consumer.close()
                
                self.kafka_producer.flush()
                self.kafka_producer.close()

    # Save the statistics
    def save_loss_plot(self, train_losses, val_losses, round_number):
        epoch = range(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epoch, train_losses, label='Training Loss')
        plt.plot(epoch, val_losses, label='Validation Loss')
        plt.title(f'Traffic Light {self.light_id} - Round {round_number}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        directory = self.create_directory_for_light('plots', 'trafffic_light')
        name = os.path.join(directory, f"plot_for_round_{round_number}.png")
        plt.savefig(name)

        directory = self.create_directory_for_light(
            'loss_csvs', 'traffic_light')
        name = name = os.path.join(
            directory, f"losses_for_round_{round_number}.csv")
        self.write_to_csv(name, train_losses, val_losses)

        plt.close()  

    # Create directory for the statistics
    def create_directory_for_light(self, directory, name):

        name = os.path.join(directory, f"{name}_{self.light_id}")

        if not os.path.exists(name):
            os.makedirs(name)
        return name

    # Write the statistics to file
    def write_to_csv(self, name, train_losses, val_losses):

        with open(name, mode='a', newline='') as file:
            write = csv.writer(file)

            for train_loss, val_loss in zip(train_losses, val_losses):
                write.writerow(
                    [f"train loss:{train_loss}", f"evaluation loss:{val_loss}"])
                

# Send the selected nn model to server
def initiate_comms(option):
    kafka_producer=KafkaProducer(bootstrap_servers='localhost:9092')

    kafka_producer.send('handshake',option.encode('utf-8'))
    kafka_producer.flush()
    kafka_producer.close()    
    print("----Started communication with server----")

# Start the clients
def start_client_process(light_id, file, partition,option):
    client = TrafficLight(light_id, file,option)
    client.initialize_nn(partition=partition)


if __name__ == "__main__":

    # Folder management
    if os.path.exists('plots'):
        shutil.rmtree('plots')

    if os.path.exists('loss_csvs'):
        shutil.rmtree('loss_csvs')


    os.makedirs('loss_csvs')
    os.makedirs('plots')

    # Select nn model
    while True:
        
        print("Select a neural network to train:")
        print("1) Simple Deep Neural Network")
        print("2) Gated Recurrent unit (GRU)")
        option=input("Select an option: ")

        if option == "1" or  option == "2":
            break
        else:
            print("You didn't put correct input")
        
        print("---------------------------------")
    
    initiate_comms(option) 

    threads = [  # Each thread represents the parallel computation in real time of each traffic light
        threading.Thread(target=start_client_process, args=(
            'I1', 'datasets/data_for_I1.pkl', 0, option)),
        threading.Thread(target=start_client_process, args=(
            'I2', 'datasets/data_for_I2.pkl', 1, option)),
        threading.Thread(target=start_client_process, args=(
            'I3', 'datasets/data_for_I3.pkl', 2, option)),
        threading.Thread(target=start_client_process, args=(
            'I4', 'datasets/data_for_I4.pkl', 3, option))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
