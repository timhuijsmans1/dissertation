import quapy as qp
import os
import pickle
import numpy as np

from quapy.method.aggregative import SVMNKLD, SVMQ, SVMRAE

def train_model(data, model_output_path):
    qp.environ['SVMPERF_HOME'] = '../svm_perf_quantification'
    model = SVMRAE()
    model.fit(data.training)
    print("fitted model")
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    print("saved model")

if __name__ == "__main__":
    TRAIN_UNION_PATH = './data/train_union.dat'
    TRAIN_EMOTICON_PATH = './data/train_emoticon.dat'
    UNION_MODEL_NAME = "./SVMRAE_union.pkl"
    EMOTICON_MODEL_NAME = "./SVMRAE_emoticon.pkl"
    MOCK_TEST_FILE = "./data/2022-02-02_balanced.dat"

    union_data = qp.data.Dataset.load(TRAIN_UNION_PATH, MOCK_TEST_FILE, qp.data.reader.from_sparse)
    emoticon_data = qp.data.Dataset.load(TRAIN_EMOTICON_PATH, MOCK_TEST_FILE, qp.data.reader.from_sparse)
    print(emoticon_data.training.instances.shape)
    train_model(union_data, UNION_MODEL_NAME)
    train_model(emoticon_data, EMOTICON_MODEL_NAME)

