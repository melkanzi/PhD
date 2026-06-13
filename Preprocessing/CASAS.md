# 1. Install required processing libraries
!pip install pandas numpy -q

import pandas as pd
import numpy as np
import os
from google.colab import drive
# import zipfile # Removed as per user request
from google.colab import files # Keep files for downloading at the end

# --- Mount Google Drive ---
print("--- Step 1: Mounting Google Drive ---")
drive.mount('/content/drive')

# --- Set path to CASAS dataset folder in Google Drive ---
# Assuming the CASAS folder with CSVs is in the root of My Drive.
# Adjust this path if your 'CASAS' folder is nested differently.
extract_path = '/content/drive/MyDrive/Datasets/CASAS' # Directory containing your CSV files

if not os.path.exists(extract_path):
    raise FileNotFoundError(f"The folder '{extract_path}' does not exist. Please ensure it's present in your Google Drive and contains CSV files.")
else:
    print(f"Reading CSV files from '{extract_path}'.")

# --- Load all CSV files from the specified directory ---
all_files = [os.path.join(extract_path, f) for f in os.listdir(extract_path) if f.endswith('.csv')]
if not all_files:
    raise FileNotFoundError(f"No CSV files found in '{extract_path}'. Please ensure the folder contains CSVs.")

print(f"Found {len(all_files)} CSV files in '{extract_path}'. Loading them into a single DataFrame.")

df_list = []
# Define common column names based on typical CASAS format: Date, Time, SensorID, State
column_names = ['date', 'time', 'sensor', 'message']

for file_path in all_files:
    try:
        # Attempt to read with header first, as in the original code's assumption
        df_single_file = pd.read_csv(file_path)

        # Check if the expected columns are present. If not, try reading without header
        # and assume space-separated values as per 'raw CASAS text file' description.
        if not all(col in df_single_file.columns for col in column_names):
            print(f"Warning: Expected columns ('date', 'time', 'sensor', 'message') not found in {file_path} with header. Trying without header, space-separated.")
            # Read without header, using regex for space/tab separation
            df_single_file = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python')

            # Assign column names if enough columns are present
            if df_single_file.shape[1] >= len(column_names):
                df_single_file.columns = column_names + [f'extra_{i}' for i in range(df_single_file.shape[1] - len(column_names))]
                df_single_file = df_single_file[column_names] # Keep only the relevant columns
            else:
                raise ValueError(f"Not enough columns in {file_path} after reading without header. Expected at least {len(column_names)}, got {df_single_file.shape[1]}.")

        df_list.append(df_single_file)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

if not df_list:
    raise ValueError("No dataframes were successfully loaded from the CSV files. Please check the dataset format.")

df_events_raw = pd.concat(df_list, ignore_index=True)

# --- Initial Parsing and Cleaning ---
print("\n--- Step 2: Initial Parsing and Cleaning of Raw Event Data ---")

# Use format='mixed' to handle inconsistent time data formats (some with milliseconds, some without)
df_events_raw['timestamp'] = pd.to_datetime(df_events_raw['date'] + ' ' + df_events_raw['time'], format='mixed')
df_events_raw = df_events_raw.rename(columns={'sensor': 'sensor_id', 'message': 'state'})
df_events_raw = df_events_raw[['timestamp', 'sensor_id', 'state']]

# Ensure chronological sorting for accurate sliding lookbacks
df_events_raw = df_events_raw.sort_values(by='timestamp').reset_index(drop=True)
# Convert timestamps into absolute elapsed seconds for mathematical deltas
df_events_raw['elapsed_sec'] = (df_events_raw['timestamp'] - df_events_raw['timestamp'].iloc[0]).dt.total_seconds()

print(f"Successfully parsed {len(df_events_raw)} chronological ambient events from all files.")

# --- PIPELINE CONFIGURATION ---
LOOKBACK_WINDOW_SEC = 10.0
FEATURE_HOUR_WINDOW = 3600.0  # Rolling 1 hour in seconds

# --- Chronological Train-Test Split ---
print("\n--- Step 2.5: Chronological Train-Test Split (70% Train, 30% Test) ---")
split_point = int(len(df_events_raw) * 0.7)
df_train = df_events_raw.iloc[:split_point].copy()
df_test = df_events_raw.iloc[split_point:].copy()

print(f"Train set size: {len(df_train)} events ({len(df_train) / len(df_events_raw):.2%})")
print(f"Test set size: {len(df_test)} events ({len(df_test) / len(df_events_raw):.2%})")

# -------------------------------------------------------------------------
# Refactored Function for CASAS Data Processing
# -------------------------------------------------------------------------

def process_casas_data(df, lookback_window_sec, feature_hour_window, dataset_name=""):
    """
    Processes a CASAS event DataFrame to extract nodes, edges, and behavioral features.

    Args:
        df (pd.DataFrame): The input DataFrame of CASAS events.
        lookback_window_sec (float): The lookback window for edge traversal in seconds.
        feature_hour_window (float): The rolling window for behavioral features in seconds.
        dataset_name (str): Name of the dataset (e.g., "Train", "Test") for printing.

    Returns:
        tuple: (X_CASAS, edge_matrix, global_node_registry, num_nodes)
    """
    print(f"\n--- Processing {dataset_name} Dataset ---")

    if df.empty:
        print(f"Warning: {dataset_name} DataFrame is empty. Returning empty matrices.")
        return np.zeros((0, 4)), np.zeros((0, 3)), {}, 0

    # Step 3: Node Extraction (\mathcal{V})
    print(f"--- {dataset_name} Step 3: Extracting Alphanumeric Hardware Node Registry ---")
    unique_sensors = df['sensor_id'].unique()
    global_node_registry = {sensor: idx for idx, sensor in enumerate(unique_sensors)}
    num_nodes = len(global_node_registry)

    print(f"Total unique ambient physical vertices registered (|V|) for {dataset_name}: {num_nodes}")
    print(f"Sensor Token Sample Map for {dataset_name}: {list(global_node_registry.items())[:5]}")

    # Step 4: Spatial-Temporal Edge Traversal & Self-Loop Debouncing
    print(f"--- {dataset_name} Step 4: Tracking Human Traversal & Debouncing Fake Self-Loops ---")

    edge_weights = {}  # Map: (Source_Idx, Dst_Idx) -> Count/Frequency

    # Iterate through the sequence starting at the second event
    if len(df) > 1:
        for t in range(1, len(df)):
            current_event = df.iloc[t]
            current_sensor = current_event['sensor_id']
            current_time = current_event['elapsed_sec']
            current_idx = global_node_registry[current_sensor]

            # Track backward up to historical ancestor frames
            for k in range(t - 1, -1, -1):
                prev_event = df.iloc[k]
                prev_sensor = prev_event['sensor_id']
                prev_time = prev_event['elapsed_sec']
                prev_idx = global_node_registry[prev_sensor]

                time_delta = current_time - prev_time

                # Guard: Break if we have moved beyond our lookback threshold
                if time_delta > lookback_window_sec:
                    break

                # Self-Loop Debouncing (Ignore repeated immediate firing of the same sensor)
                if prev_sensor == current_sensor:
                    continue

                # Valid spatial-temporal link captured
                if 0 < time_delta <= lookback_window_sec:
                    edge_pair = (prev_idx, current_idx)
                    edge_weights[edge_pair] = edge_weights.get(edge_pair, 0) + 1
                    break # Found the most immediate valid historical spatial ancestor, move to next t

    # Convert our tracked dictionary into a standard GNN tensor format
    edge_list = [[src, dst, weight] for (src, dst), weight in edge_weights.items()]
    edge_matrix = np.array(edge_list) if len(edge_list) > 0 else np.zeros((0, 3))

    print(f"Generated {len(edge_matrix)} active spatial transition paths for {dataset_name}.")

    # Step 5: Behavioral Feature Synthesis
    print(f"--- {dataset_name} Step 5: Extracting Rolling Behavioral Feature Space Attributes ---")

    # Initialize feature accumulators for each unique hardware token row
    frequency_counts = np.zeros(num_nodes)
    duration_states = np.zeros(num_nodes)
    analog_means = np.zeros(num_nodes)
    analog_stds = np.zeros(num_nodes)

    # Keep track of active durations for sensors that utilize explicit ON/OFF toggles
    sensor_last_on_time = {}

    # Compute behavioral traits over operational windows
    max_time = df['elapsed_sec'].max() if not df.empty else 0.0

    for idx, sensor in enumerate(unique_sensors):
        node_idx = global_node_registry[sensor]
        sensor_subset = df[df['sensor_id'] == sensor]

        # 1. Frequency Count (Trips over the final sliding hour of capture window)
        # For the test set, the 'one_hour_cutoff' should still be relative to its own max_time
        one_hour_cutoff = max_time - feature_hour_window
        recent_trips = sensor_subset[sensor_subset['elapsed_sec'] >= one_hour_cutoff]
        frequency_counts[node_idx] = len(recent_trips)

        # 2. Duration State Calculation
        total_duration = 0.0
        for _, row in sensor_subset.iterrows():
            state_val = str(row['state']).upper()
            if "ON" in state_val or "OPEN" in state_val:
                sensor_last_on_time[sensor] = row['elapsed_sec']
            elif ("OFF" in state_val or "CLOSE" in state_val) and sensor in sensor_last_on_time:
                total_duration += (row['elapsed_sec'] - sensor_last_on_time[sensor])
                del sensor_last_on_time[sensor] # Reset latch state

        duration_states[node_idx] = total_duration

        # 3. Value Trajectory (Process analog scales for float output fields like temperature sensors)
        try:
            analog_values = pd.to_numeric(sensor_subset['state'], errors='coerce').dropna().values
            if len(analog_values) > 0:
                analog_means[node_idx] = np.mean(analog_values)
                analog_stds[node_idx] = np.std(analog_values)
        except Exception:
            pass # Stays zero if sensor output profile is purely categorical/binary text

    # Step 6: Final Matrix Formulation (X_CASAS)
    print(f"--- {dataset_name} Step 6: Engineering Consolidated Node Feature Matrix Tensors ---")

    X_CASAS = np.column_stack([
        frequency_counts,
        duration_states,
        analog_means,
        analog_stds
    ])

    print(f"Final Graph Tensors Shape Profile for {dataset_name}:")
    print(f"  Feature Matrix (Nodes x 4 Attributes): {X_CASAS.shape}")
    print(f"  Spatial Edge Matrix [Src, Dst, Transition_Weight]: {edge_matrix.shape}")

    return X_CASAS, edge_matrix, global_node_registry, num_nodes


# -------------------------------------------------------------------------
# Apply processing to Train and Test datasets
# -------------------------------------------------------------------------
X_CASAS_train, edge_CASAS_train, global_node_registry_train, num_nodes_train = process_casas_data(
    df_train, LOOKBACK_WINDOW_SEC, FEATURE_HOUR_WINDOW, dataset_name="Train"
)

X_CASAS_test, edge_CASAS_test, global_node_registry_test, num_nodes_test = process_casas_data(
    df_test, LOOKBACK_WINDOW_SEC, FEATURE_HOUR_WINDOW, dataset_name="Test"
)

# Save processed matrices
print("\n--- Step 7: Saving and Downloading Processed Tensors ---")
np.save('X_CASAS_train.npy', X_CASAS_train)
np.save('edge_CASAS_train.npy', edge_CASAS_train)
np.save('X_CASAS_test.npy', X_CASAS_test)
np.save('edge_CASAS_test.npy', edge_CASAS_test)

print("Downloading generated files...")
files.download('X_CASAS_train.npy')
files.download('edge_CASAS_train.npy')
files.download('X_CASAS_test.npy')
files.download('edge_CASAS_test.npy')
print("Processing Finished Successfully!")

