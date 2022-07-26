import json
import os

def class_balance_counter(path):
    label_counts = {'positive': 0, 'negative': 0}
    with open(path, 'r') as f:
        for line in f:
            if int(line.split(" ")[0]) == 1:
                label_counts['positive'] +=  1
            if int(line.split(" ")[0]) == -1:
                label_counts['negative'] += 1
    
    lowest_count = min(label_counts.values())
    return lowest_count

def class_rebalancer(lowest_count, input_path, output_path):
    neg_count = 0
    pos_count = 0
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            label = int(line.split(" ")[0])
            if label == 1:
                if pos_count < lowest_count:
                    f_out.write(line)
                    pos_count += 1
            if label == -1:
                if neg_count < lowest_count:
                    f_out.write(line)
                    neg_count += 1

def output_path_generator(input_path):
    file_name = input_path.split("/")[-1]
    
    path = input_path.strip(file_name)
    file_name_stripped = file_name.split(".")[0]
    output_path = "../" + os.path.join(path, file_name_stripped + "_balanced.dat")
    return output_path

if __name__ == "__main__":
    INSTANCE_PATH = '../data/train_test_data/weekly_split/instances/test/2022-02-02.dat'
    OUTPUT_PATH = output_path_generator(INSTANCE_PATH)
    lowest_count = class_balance_counter(INSTANCE_PATH)
    class_rebalancer(lowest_count, INSTANCE_PATH, OUTPUT_PATH)