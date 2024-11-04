
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import multiprocessing
import torch
import random
import itertools
import pickle
import neural_algorithms as deep
from uxsim import World
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('Agg')

class Simulation:
    def __init__(self, use_nn, seed, mode):
        self.use_nn = use_nn
        self.reset(seed)

        if use_nn:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            # Load the 4 models trained by the clients
            if mode == "Simple Deep Neural Network":
                self.model = deep.DeepNN().to(self.device)
                self.model.load_state_dict(torch.load(
                    f'NN_models/global_model_deepNN.pth', map_location=self.device))
            else:
                self.model = deep.GRU().to(self.device)
                self.model.load_state_dict(torch.load(
                    f'NN_models/global_model_gru.pth', map_location=self.device))

            self.model.eval()

            # Load the global scalers
            with open('scalers/densities_scaler.pkl', 'rb') as f:
                self.scaler_X = pickle.load(f)

            with open('scalers/timings_scaler.pkl', 'rb') as f:
                self.scaler_y = pickle.load(f)

    def reset(self, seed):
        """
        Set up the traffic simulation environment and network.
        """
        W = World(
            name="",
            deltan=5,
            tmax=3600,
            print_mode=0, save_mode=1, show_mode=0,
            random_seed=seed,
            duo_update_time=600
        )

        # random.seed(seed)

        # Network definition
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
        for n1, n2 in [[W1, self.I1], [self.I1, self.I2], [self.I2, E1], [W2, self.I3], [self.I3, self.I4],
                       [self.I4, E2]]:
            W.addLink(n1.name + n2.name, n1, n2, length=1000,
                      free_flow_speed=30, jam_density=0.2, signal_group=0)
            W.addLink(n2.name + n1.name, n2, n1, length=1000,
                      free_flow_speed=30, jam_density=0.2, signal_group=0)

        # N <-> S direction: signal group 1
        for n1, n2 in [[N1, self.I1], [self.I1, self.I3], [self.I3, S1], [N2, self.I2], [self.I2, self.I4],
                       [self.I4, S2]]:
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

        self.W = W

    def get_density(self, intersection):
        densities = []
        for link in intersection.inlinks.values():
            if hasattr(link, 'density'):
                densities.append(link.density)
            else:
                densities.append(0)
        return densities

    def lights_timing(self):
        # Update timings for each intersection using its own model
        for intersection in [self.I1, self.I2, self.I3, self.I4]:
            model = self.model

            densities = self.get_density(intersection)
            # Normalize the densities using the global scaler
            densities_normalized = self.scaler_X.transform([densities])

            densities_tensor = torch.tensor(
                densities_normalized, dtype=torch.float32).to(self.device)

            # Get the model's output
            with torch.no_grad():
                timings_normalized = model(densities_tensor)

            # convert the tensors to a lsit
            timings_normalized = timings_normalized.cpu().tolist()

            # Denormalize the timings
            timings = self.scaler_y.inverse_transform(timings_normalized)

            # update the lights based on the prediction of the nn
            self.set_light_timings(intersection, timings[0])

    def lights_timing(self):
        # Update timings for each intersection using its own model
        for intersection in [self.I1, self.I2, self.I3, self.I4]:
            model = self.model

            densities = self.get_density(intersection)
            # Normalize the densities using the global scaler
            densities_normalized = self.scaler_X.transform([densities])

            densities_tensor = torch.tensor(
                densities_normalized, dtype=torch.float32).to(self.device)

            # Get the model's output
            with torch.no_grad():
                timings_normalized = model(densities_tensor)

            # convert the tensors to a lsit
            timings_normalized = timings_normalized.cpu().tolist()

            # Denormalize the timings
            timings = self.scaler_y.inverse_transform(timings_normalized)

            # update the lights based on the prediction of the nn
            self.set_light_timings(intersection, timings[0])

    def set_light_timings(self, light, timings):

        green_time = timings

        # Adjust green timing for each road section
        light.signal[0] = round(green_time[0])
        light.signal[1] = round(green_time[1])

    def update_lights(self):
        if self.use_nn:
            self.lights_timing()

    def run_simulation(self, operation_timestep_width=100):
        if self.use_nn:
            self.W.exec_simulation(duration_t=operation_timestep_width)
            self.update_lights()
        else:
            self.W.exec_simulation()

    def simple_stats_to_json(self):
        simple_json = {}

        simple_json["average_speed"] = self.W.analyzer.average_speed
        simple_json["completed_trips"] = self.W.analyzer.trip_completed
        simple_json["total_trips"] = self.W.analyzer.trip_all
        simple_json["delay"] = self.W.analyzer.average_delay
        simple_json["delay_ratio"] = simple_json["delay"] / \
            self.W.analyzer.average_travel_time
        # simple_json['total_distance_traveled']=self.W.analyzer.total_distance_traveled

        link_info = self.W.analyzer.link_to_pandas()

        average_trafiic_volume = link_info["traffic_volume"].mean()

        simple_json["average_traffic_volume"] = average_trafiic_volume

        return simple_json


def get_network(mode):
    sim_net = Simulation(False,0,mode)
    sim_net.reset(0)
    sim_net.W.show_network()


def init_simulations(seed,mode):
    sim = Simulation(True, seed,mode)

    while sim.W.check_simulation_ongoing():
        sim.run_simulation()

    # compare a simulation that is controlled by the nn with a one that isnt(same world values,because they are generated by the same seed)
    sim_no_nn = Simulation(False, seed,mode)
    while sim_no_nn.W.check_simulation_ongoing():
        sim_no_nn.run_simulation()

    return sim, sim_no_nn

# Simulation with the selected nn model
def init_simulation_with_nn(seed,mode):
    sim = Simulation(True, seed, mode)

    while sim.W.check_simulation_ongoing():
        sim.run_simulation()

    return sim

# Simulation without nn model
def init_simulation_without_nn(seed,mode):
    sim_no_nn = Simulation(False, seed,mode)

    while sim_no_nn.W.check_simulation_ongoing():
        sim_no_nn.run_simulation()

    return sim_no_nn


def run_simulation_pair(args):
    seed,mode=args
    # Run simulation with NN
    sim = Simulation(True, seed, mode)
    while sim.W.check_simulation_ongoing():
        sim.run_simulation()

    # Run simulation without NN
    sim_no_nn = Simulation(False, seed, mode)
    while sim_no_nn.W.check_simulation_ongoing():
        sim_no_nn.run_simulation()

    # Extract average speeds from both simulations
    sim_stats = sim.simple_stats_to_json()  # Contains stats like "average_speed"
    sim_no_nn_stats = sim_no_nn.simple_stats_to_json()

    # Return a tuple of (average_speed with NN, average_speed without NN)
    return sim_stats["average_speed"], sim_no_nn_stats["average_speed"]


def run_for_many_seeds(mode):
    num_simulations = 100
    seeds = range(1, num_simulations + 1)

    # Parallel execution of simulations using multiprocessing
    # Get the number of available CPU cores
    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        args=[(seed,mode) for seed in seeds]
        results = list(
            tqdm(pool.imap(run_simulation_pair, args), total=num_simulations))

    # Separate the results into two lists: one for sim and one for sim_no_nn
    sim_speeds = [result[0]
                  for result in results]  # Average speeds for sim (with NN)
    # Average speeds for sim_no_nn (without NN)
    sim_no_nn_speeds = [result[1] for result in results]

    # Plotting the results
    plt.figure(figsize=(10, 10)) 

    plt.plot(seeds, sim_speeds, label=f'With {mode}', color='blue', marker='o')
    plt.plot(seeds, sim_no_nn_speeds, label='Without NN', color='red', marker='x')
    plt.xlabel('Seed')
    plt.ylabel('Average Speed (m/s)')
    plt.title('Comparison of Average Speed: With NN vs Without NN')
    plt.legend()

    name = os.path.join('out', f"run_for_many_seeds.png")
    plt.savefig(name)  
    plt.close()
