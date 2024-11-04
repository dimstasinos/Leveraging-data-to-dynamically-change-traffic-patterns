import torch
import torch.nn as nn
import pickle
import neural_algorithms as deep
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_pickle(client_id):
    """Load the test data for a specific client."""
    data_inputs = []
    data_targets = []

    # Load the test data for the client
    with open(f'datasets/data_for_{client_id}.pkl', 'rb') as f:
        while True:
            try:
                densities, timings = pickle.load(f)
                data_inputs.append(densities)
                data_targets.append(timings)
            except EOFError:
                break

    return np.array(data_inputs), np.array(data_targets)


def evaluate_model(client_id, model_path, densities_scaler, timings_scaler, nn_model, device):
    """Evaluate the model for a specific client."""

    # Load the model for the client
    model = nn_model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load and normalize the test data
    densities, timings = load_pickle(client_id)
    densities_normalized = densities_scaler.transform(densities)

    # convert the densities to tensors to load them to the model
    densities_tensor = torch.tensor(
        densities_normalized, dtype=torch.float32).to(device)

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(densities_tensor).cpu().numpy()

    # Denormalize the predictions
    predictions_denorm = timings_scaler.inverse_transform(predictions)

    # Compute the mse and mae to see the model's accuracy
    mse = mean_squared_error(timings, predictions_denorm)
    mae = mean_absolute_error(timings, predictions_denorm)

    print(f"Client {client_id} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot a comparison between predicted and actual values for visual analysis
    num_samples_to_plot = 1000
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_denorm[:num_samples_to_plot,
             0], label='Predicted Green Time')
    plt.plot(timings[:num_samples_to_plot, 0], label='Actual Green Time')
    plt.legend()
    plt.title(f"Client {client_id}: Predicted vs. Actual Green Timing")
    plt.xlabel("Sample")
    plt.ylabel("Time (seconds)")
    plt.show()

    return predictions_denorm, timings


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the scalers
    with open('scalers/densities_scaler.pkl', 'rb') as f:
        densities_scaler = pickle.load(f)

    with open('scalers/timings_scaler.pkl', 'rb') as f:
        timings_scaler = pickle.load(f)

    # Select a nn to evaluate
    while True:
        print("Select a neural network to evaluate:")
        print("1) Simple Deep Neural Network")
        print("2) Gated Recurrent unit (GRU)")
        option = input("Select an option: ")

        if option == "1" or option == "2":
            break
        else:
            print("You didn't put correct input")

        print("---------------------------------")

    if option == "1":
        model_name = "deepNN"
        model = deep.DeepNN().to(device)

    else:
        model_name = "gru"
        model = deep.GRU().to(device)

    # Evaluate each model
    clients = ['I1', 'I2', 'I3', 'I4']
    for client_id in clients:
        print(f"\nEvaluating model for {client_id}...")
        # the global model that the server saves
        model_path = f'NN_models/global_model_{model_name}.pth'
        evaluate_model(client_id, model_path, densities_scaler,
                       timings_scaler, model, device)


if __name__ == "__main__":
    main()
