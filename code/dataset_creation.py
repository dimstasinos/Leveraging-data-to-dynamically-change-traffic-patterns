import itertools
import os
import pickle
import random
from tqdm import tqdm
from uxsim import World
import multiprocessing
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class TrafficSim:
    def __init__(self):
        """
        Initialize the traffic simulation with 4 signalized intersections.
        """
        self.W = None
        self.I1_data = []
        self.I2_data = []
        self.I3_data = []
        self.I4_data = []

    def reset(self, seed):
        """
        Settting up main evnviroment 
        """
        self.seed = seed  # Use None for random or set a specific seed
        W = World(
            name="",
            deltan=5,
            tmax=3600,
            print_mode=0, save_mode=0, show_mode=1,
            random_seed=seed,
            duo_update_time=600
        )

        # Define the intersections with consistent ordering
        # Default timing, will adjust later
        self.I1 = W.addNode("I1", 0, 0, signal=[60, 60])
        self.I2 = W.addNode("I2", 1, 0, signal=[60, 60])
        self.I3 = W.addNode("I3", 0, -1, signal=[60, 60])
        self.I4 = W.addNode("I4", 1, -1, signal=[60, 60])

        W1 = W.addNode("W1", -1, 0)
        W2 = W.addNode("W2", -1, -1)
        E1 = W.addNode("E1", 2, 0)
        E2 = W.addNode("E2", 2, -1)
        N1 = W.addNode("N1", 0, 1)
        N2 = W.addNode("N2", 1, 1)
        S1 = W.addNode("S1", 0, -2)
        S2 = W.addNode("S2", 1, -2)

        # E <-> W direction: signal group 0
        for n1, n2 in [[W1, self.I1], [self.I1, self.I2], [self.I2, E1],
                       [W2, self.I3], [self.I3, self.I4], [self.I4, E2]]:
            W.addLink(n1.name + n2.name, n1, n2, length=1000,
                      free_flow_speed=30, jam_density=0.2, signal_group=0)
            W.addLink(n2.name + n1.name, n2, n1, length=1000,
                      free_flow_speed=30, jam_density=0.2, signal_group=0)

        # N <-> S direction: signal group 1
        for n1, n2 in [[N1, self.I1], [self.I1, self.I3], [self.I3, S1],
                       [N2, self.I2], [self.I2, self.I4], [self.I4, S2]]:
            W.addLink(n1.name + n2.name, n1, n2, length=1000,
                      free_flow_speed=30, jam_density=0.2, signal_group=1)
            W.addLink(n2.name + n1.name, n2, n1, length=1000,
                      free_flow_speed=30, jam_density=0.2, signal_group=1)

        # Random demand definition
        dt = 30
        demand = 0.22
        for n1, n2 in itertools.permutations([W1, W2, E1, E2, N1, N2, S1, S2], 2):
            for t in range(0, 3600, dt):
                W.adddemand(n1, n2, t, t + dt, random.uniform(0, demand))

        # Store the simulation world for later use
        self.W = W

        # Prepare to store the simulation data
        self.I1_data = []
        self.I2_data = []
        self.I3_data = []
        self.I4_data = []

    def get_densities_of_light(self, traffic_light):
        #get the densities of each road that is  connected in the traffic light in a list
        incoming_links = sorted(
            traffic_light.inlinks.values(), key=lambda l: l.name)
        densities = [l.density for l in incoming_links]
        return densities

    def adjust_signals(self, densities):
        """
        based on the densities on each road section, we adjust the time signals to "relieve" the busiest section,
        that data is passed on the neural networks ,so they can learn based on the density of the road, what timings help the reduce the densities
        """
        #  More density => more green light time

        NS = densities[2] + densities[3]
        EW = densities[0] + densities[1]

        total_density = NS + EW

        if total_density > 0:
            # Calculate the green time ratio for North-South and East-West based on their total densities
            green_ratio_NS = NS / total_density
            green_ratio_EW = EW / total_density
        else:
            # If no traffic, give equal green time to both directions
            green_ratio_NS = green_ratio_EW = 0.5

        # normalize the values in seconds
        
        green_time_NS = int(green_ratio_NS * 100)
       
        green_time_EW = int(green_ratio_EW * 100)

        return [green_time_EW, green_time_NS]

    def run_simulation(self):
        operation_timestep_width = 150
        dataset = []


        while self.W.check_simulation_ongoing():
            #call the above function to get the data needed
            i1 = self.get_densities_of_light(self.I1)
            i2 = self.get_densities_of_light(self.I2)
            i3 = self.get_densities_of_light(self.I3)
            i4 = self.get_densities_of_light(self.I4)

           
            self.I1_data = i1
            self.I2_data = i2
            self.I3_data = i3
            self.I4_data = i4

           
            I1_green_times = self.adjust_signals(i1)
            I2_green_times = self.adjust_signals(i2)
            I3_green_times = self.adjust_signals(i3)
            I4_green_times = self.adjust_signals(i4)

            # change the timings of the signals in order to sample the effect on the change of the road densities
            #and store the "otpimal" timings for the densities
            self.I1.signal[0] = I1_green_times[0]
            self.I1.signal[1] = I1_green_times[1]
            self.I2.signal[0] = I2_green_times[0]
            self.I2.signal[1] = I2_green_times[1]
            self.I3.signal[0] = I3_green_times[0]
            self.I3.signal[1] = I3_green_times[1]
            self.I4.signal[0] = I4_green_times[0]
            self.I4.signal[1] = I4_green_times[1]

            #make a simple object to capure the data
            row = {
                'I1_data': self.I1_data, 'I2_data': self.I2_data,
                'I3_data': self.I3_data, 'I4_data': self.I4_data,
                'I1_green': self.I1.signal[0], 'I1_red':  self.I1.signal[1],
                'I2_green': self.I2.signal[0], 'I2_red': self.I2.signal[1],
                'I3_green': self.I3.signal[0], 'I3_red': self.I3.signal[1],
                'I4_green': self.I4.signal[0], 'I4_red': self.I4.signal[1]
            }
            #append each row to the complete dataser
            dataset.append(row)
            # exeute the simulation for the timestep
            self.W.exec_simulation(duration_t=operation_timestep_width)
        #after the end return the complete dataset as a panda
        return pd.DataFrame(dataset)


def run_simulation_for_seed(seed):
    sim = TrafficSim()

    #run the simulation for the specific seed(more seeds more data for the nns to learn)
    sim.reset(seed)
    dataset = sim.run_simulation()

    return dataset


def write_to_dataset(dataset): #write the datasets as a big list that has the lists of densities and timings(easy template to feed to the nns)
    I1_data = [
        [row['I1_data'], [row['I1_green'], row['I1_red']]]
        for index, row in dataset.iterrows()
    ]

    I2_data = [
        [row['I2_data'], [row['I2_green'], row['I2_red']]]
        for index, row in dataset.iterrows()
    ]

    I3_data = [
        [row['I3_data'], [row['I3_green'], row['I3_red']]]
        for index, row in dataset.iterrows()
    ]

    I4_data = [
        [row['I4_data'], [row['I4_green'], row['I4_red']]]
        for index, row in dataset.iterrows()
    ]

    # Ensure the 'datasets' folder exists
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    # store them as bytes in a pickle file to optimize both the transfer and the storing of the values
    with open('datasets/data_for_I1.pkl', 'ab') as file:
        for item in I1_data:
            pickle.dump(item, file)

    with open('datasets/data_for_I1.txt', 'a') as file:
        for item in I1_data:
            file.write(f"{item}\n")

    with open('datasets/data_for_I2.pkl', 'ab') as file:
        for item in I2_data:
            pickle.dump(item, file)

    with open('datasets/data_for_I3.pkl', 'ab') as file:
        for item in I3_data:
            pickle.dump(item, file)

    with open('datasets/data_for_I4.pkl', 'ab') as file:
        for item in I4_data:
            pickle.dump(item, file)


if __name__ == "__main__":
    # remove the files if they already exist, in order to remove previous data
    for filename in ['datasets/data_for_I1.pkl', 'datasets/data_for_I2.pkl', 'datasets/data_for_I3.pkl', 'datasets/data_for_I4.pkl', 'datasets/data_for_I1.txt']:
        if os.path.exists(filename):
            os.remove(filename)

    # number of seeds,more seeds more data
    num_simulations = 200  
    seeds = [i for i in range(num_simulations)]

    # to run the programm faster we parallelize it: each thread of the pc->one simulation
    num_workers = multiprocessing.cpu_count()

    # run the programm, and a basic way to see when it ends
    with multiprocessing.Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(run_simulation_for_seed, seeds), total=num_simulations))

    # write the results of all the seeds
    for dataset in results:
        write_to_dataset(dataset)
