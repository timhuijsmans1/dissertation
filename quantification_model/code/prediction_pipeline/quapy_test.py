from .. import quapy as qp
import os
from quapy.method.aggregative import SVMQ

def custom_loader(path):
    # TODO: possibly rewrite this using np.loadtxt from .dat file
    with open(path, 'rb') as f:
        # load all data into lines as list
        lines = []
        while True:
            line = f.readline()
            if not line:
                break
            lines.append(line.decode().strip("\n").split(" "))
        
        # split data in labels and instances
        labels = []
        instances = []

        for data_line in lines:
            label, sparse_features = int(data_line[0]), data_line[1:]

            sparse_feature_dict = {}
            for sparse_feature in sparse_features:
                k,v = sparse_feature.split(":")
                sparse_feature_dict[int(k)] = float(v)

            # instantiate and append a feature list for each instance
            features = []
            total_features = 10 # TODO: find a way to not hardcode this magic number 
            for i in range(1, total_features + 1):
                feature_value = sparse_feature_dict.get(i)
                if feature_value != None:
                    features.append(feature_value)
                else:
                    features.append(0)
            
            labels.append(label)
            instances.append(features)

        return instances, labels    

def train_model(data):
    qp.environ['SVMPERF_HOME'] = '../svm_perf_quantification'
    model = SVMQ()
    model.fit(data.training)
    predicted_labels = model.classify(data.test.instances)
    print(predicted_labels)

if __name__ == "__main__":
    SEM_EVAL_16_PATH = "/Users/timhuijsmans/quapy_data/tweet_sentiment_quantification_snam/train/semeval16.train+dev.feature.txt"
    CUSTOM_PATH_TRAIN = "./test_data_tim/train.dat"
    CUSTOM_PATH_TEST = "./test_data_tim/test.dat"
    # data = qp.data.reader.from_sparse(CUSTOM_PATH_TRAIN)
    # print(data[1])
    # data = qp.data.Dataset.load(CUSTOM_PATH_TRAIN, CUSTOM_PATH_TEST, custom_loader)
    # print(data.training.labels)
    # print(data.training.instances)

    data = qp.data.Dataset.load(CUSTOM_PATH_TRAIN, CUSTOM_PATH_TEST, qp.data.reader.from_sparse)
    train_model(data)