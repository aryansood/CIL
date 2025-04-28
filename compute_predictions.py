import os
import numpy as np
import pandas as pd
import base64
import zlib
from tqdm import tqdm
from utils.constants import DATA_DIR

# Path definitions
data_root = DATA_DIR
predictions_dir = os.path.join(data_root, 'predictions')
test_list_file = os.path.join(data_root, 'test_list.txt')
output_csv = os.path.join(data_root, 'predictions.csv')

def compress_depth_values(depth_values):
    depth_bytes = ','.join(f"{x:.2f}" for x in depth_values).encode('utf-8')
    compressed = zlib.compress(depth_bytes, level=9) 

    return base64.b64encode(compressed).decode('utf-8')

def process_depth_maps():
    with open(test_list_file, 'r') as f:
        file_pairs = [line.strip().split() for line in f]
    
    ids = []
    depths_list = []
    
    for rgb_path, depth_path in tqdm(file_pairs, desc="Processing depth maps"):
        file_id = os.path.splitext(os.path.basename(depth_path))[0]
        
        depth = np.load(os.path.join(predictions_dir, depth_path))
        flattened_depth = np.round(depth.flatten(), 2)
        
        compressed_depths = compress_depth_values(flattened_depth)
        ids.append(file_id)
        depths_list.append(compressed_depths)

    df = pd.DataFrame({
        'id': ids,
        'Depths': depths_list,
    })
    
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to: {output_csv}")
    print(f"Shape of the CSV: {df.shape}")
    

if __name__ == "__main__":
    process_depth_maps() 