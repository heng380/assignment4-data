import os
import mmh3
from tqdm import tqdm
import concurrent.futures
import pickle

def exact_line_dedup(inputfiles, output_directory):
    unique_lines = {}
    for file in tqdm(inputfiles):
        file_name = os.path.basename(file)

        with open(file) as f:
            for line in f:
                line_hash = mmh3.hash(line, signed=False)
                if line_hash not in unique_lines:
                    unique_lines[line_hash] = {
                        "count": 0
                    }
                unique_lines[line_hash]["count"] += 1
    for file in tqdm(inputfiles):
        file_name = os.path.basename(file)
        with open(file) as f:
            with open(os.path.join(output_directory, file_name), "w") as f_out:
                for line in f:
                    line_hash = mmh3.hash(line, signed=False)
                    if unique_lines[line_hash]["count"] == 1:
                        f_out.write(line)
                        
                    