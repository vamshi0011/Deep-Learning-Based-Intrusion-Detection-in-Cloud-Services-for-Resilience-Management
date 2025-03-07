import pandas as pd
import numpy as np
import os

# Ensure 'data/' directory exists before saving
output_dir = "C:/Users/Vamshi/Desktop/Intrusion_detection/data"
os.makedirs(output_dir, exist_ok=True)

# Define total dataset size
TOTAL_SIZE = 1000  

# Define train-test split ratios
SPLITS = {
    "train_80": 0.8, "test_20": 0.2,
    "train_70": 0.7, "test_30": 0.3,
    "train_60": 0.6, "test_40": 0.4,
    "train_50": 0.5, "test_50": 0.5
}

# Define categorical feature values
PROTOCOLS = ['TCP', 'UDP', 'ICMP']
SERVICES = ['HTTP', 'FTP', 'SSH', 'DNS', 'SMTP']
FLAGS = ['SF', 'S0', 'REJ', 'RSTO', 'SH']

# Function to generate synthetic network traffic data
def generate_data(size):
    np.random.seed(42)  # For reproducibility
    data = {
        'duration': np.random.randint(0, 5000, size),
        'protocol_type': np.random.choice(PROTOCOLS, size),
        'service': np.random.choice(SERVICES, size),
        'flag': np.random.choice(FLAGS, size),
        'src_bytes': np.random.randint(0, 10000, size),
        'dst_bytes': np.random.randint(0, 10000, size),
        'count': np.random.randint(0, 100, size),
        'same_srv_rate': np.random.uniform(0, 1, size),
        'diff_srv_rate': np.random.uniform(0, 1, size),
        'dst_host_count': np.random.randint(1, 255, size),
        'attack_type': np.random.choice(['Normal', 'Attack'], size, p=[0.7, 0.3])  # 70% normal, 30% attack
    }
    return pd.DataFrame(data)

# Generate full dataset and shuffle
full_df = generate_data(TOTAL_SIZE)
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset and save to CSV files
start_idx = 0
for name, ratio in SPLITS.items():
    size = int(TOTAL_SIZE * ratio)
    split_df = full_df.iloc[start_idx:start_idx + size]
    file_path = f"{output_dir}/{name}.csv"
    split_df.to_csv(file_path, index=False)
    print(f"✅ {file_path} generated with {size} samples.")
    start_idx += size

print("\n🎉 All datasets have been successfully generated! 🚀")
