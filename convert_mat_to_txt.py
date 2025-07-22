#!/usr/bin/env python3
"""
Convert MATLAB .mat files to .txt format for Rust implementation
Searches for .mat files in various directories and converts them
"""

import os
import numpy as np
from scipy.io import loadmat
import sys

np.set_printoptions(precision=25, threshold=sys.maxsize, linewidth=5000)

def convert_mat_file(mat_path, txt_path):
    """Convert a single .mat file to .txt format"""
    try:
        GNBG_tmp = loadmat(mat_path)['GNBG']
        
        # Extract all parameters
        MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
        AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
        Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
        CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
        MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
        MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
        CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
        CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
        CompH = np.array(GNBG_tmp['Component_H'][0, 0])
        Mu = np.array(GNBG_tmp['Mu'][0, 0])
        Omega = np.array(GNBG_tmp['Omega'][0, 0])
        Lambda = np.array(GNBG_tmp['lambda'][0, 0])
        RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
        OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
        OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
        
        # Format as string
        contents = ""
        contents += np.array2string(MaxEvals) + "\n"
        contents += np.array2string(AcceptanceThreshold) + "\n"
        contents += np.array2string(Dimension) + "\n"
        contents += np.array2string(CompNum) + "\n"
        contents += np.array2string(MinCoordinate) + "\n"
        contents += np.array2string(MaxCoordinate) + "\n"
        contents += np.array2string(CompMinPos) + "\n"
        contents += np.array2string(CompSigma) + "\n"
        contents += np.array2string(CompH) + "\n"
        contents += np.array2string(Mu) + "\n"
        contents += np.array2string(Omega) + "\n"
        contents += np.array2string(Lambda) + "\n"
        contents += np.array2string(RotationMatrix) + "\n"
        contents += np.array2string(OptimumValue) + "\n"
        contents += np.array2string(OptimumPosition) + "\n"
        
        # Clean up formatting
        contents = contents.replace("[", "")
        contents = contents.replace("]", "")
        contents = contents.replace("\n\n", "\n")
        
        # Write to file
        with open(txt_path, 'w') as f:
            f.write(contents)
        
        return True
    except Exception as e:
        print(f"  Error converting {mat_path}: {e}")
        return False

def find_and_convert_mat_files():
    """Find all .mat files and convert them to .txt in the same directory"""
    # Directories to search
    search_dirs = [
        ".",
        "Python_Implementation/GNBG_Instances.Python-main",
        "MATLAB_Implementation/GNBG II- Instance.MATLAB",
        "C_Implementation/GNBG-Instance-C-main",
    ]
    
    converted_count = 0
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        print(f"\nSearching in {search_dir}...")
        
        # Find all f*.mat files
        mat_files = []
        for i in range(1, 25):
            mat_file = os.path.join(search_dir, f"f{i}.mat")
            if os.path.exists(mat_file):
                mat_files.append((i, mat_file))
        
        if mat_files:
            print(f"Found {len(mat_files)} .mat files")
            
            for idx, mat_path in mat_files:
                txt_path = mat_path.replace('.mat', '.txt')
                
                # Skip if already converted
                if os.path.exists(txt_path):
                    print(f"  f{idx}.txt already exists, skipping")
                    continue
                
                print(f"  Converting f{idx}.mat -> f{idx}.txt", end="... ")
                if convert_mat_file(mat_path, txt_path):
                    print("OK")
                    converted_count += 1
                else:
                    print("FAILED")
    
    return converted_count

if __name__ == "__main__":
    print("GNBG .mat to .txt converter")
    print("=" * 40)
    
    try:
        import scipy
        print(f"Using scipy version: {scipy.__version__}")
    except ImportError:
        print("ERROR: scipy not installed. Please run: pip install scipy")
        sys.exit(1)
    
    converted = find_and_convert_mat_files()
    
    print(f"\nConversion complete! Converted {converted} files.")
    
    if converted == 0:
        print("\nNo new files to convert. All .mat files already have corresponding .txt files.")
    else:
        print(f"\nSuccessfully converted {converted} .mat files to .txt format.")