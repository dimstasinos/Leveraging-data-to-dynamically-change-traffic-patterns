# create_global_scaler.py

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


all_data_inputs = []
all_data_targets = []

# For each client, load their data
for client_id in ['I1', 'I2', 'I3', 'I4']:
    
    client_data = []
    with open(f'datasets/data_for_{client_id}.pkl', 'rb') as f:
        while True:
            try:
                client_data.append(pickle.load(f))
            except EOFError:
                break
     #append all the values in order to make scalers based on all the traffic lights           
    for row in client_data:
        densities, timings = row
        all_data_inputs.append(densities)
        all_data_targets.append(timings)


all_data_inputs = np.array(all_data_inputs)
all_data_targets = np.array(all_data_targets)


densities_scaler = StandardScaler()
timings_scaler = StandardScaler()

densities_scaler.fit(all_data_inputs)
timings_scaler.fit(all_data_targets)

# Save the global scalers
with open('scalers/densities_scaler.pkl', 'wb') as f:
    pickle.dump(densities_scaler, f)

with open('scalers/timings_scaler.pkl', 'wb') as f:
    pickle.dump(timings_scaler, f)

print("Global scalers created and saved as 'densities_scaler.pkl' and 'timings_scaler.pkl'.")
