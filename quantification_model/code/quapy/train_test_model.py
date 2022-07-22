import quapy as qp
import os
import pickle
import numpy as np

from quapy.method.aggregative import SVMNKLD, OneVsAll
from quapy import functional

def train_model(data):
    qp.environ['SVMPERF_HOME'] = '../svm_perf_quantification'
    model = SVMNKLD()
    model.fit(data.training)
    print("fitted model")
    with open('./model_SVMNKLD_balanced.pkl', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    print("saved model")

def prevalence_from_labels(labels):
    uniques, counts = np.unique(labels, return_counts=True)
    class_2_prevalence = dict(zip(uniques, counts / counts.sum()))
    return class_2_prevalence

def predict(data):
    with open('./model_SVMNKLD_balanced.pkl', 'rb') as f:
        model = pickle.load(f)
    predicted_labels = model.classify(data.test.instances)
    predicted_prevalence = prevalence_from_labels(predicted_labels)

    true_labels = data.test.labels
    true_prevalence = prevalence_from_labels(true_labels)

    print('true ', true_prevalence)
    print('predicted', predicted_prevalence)

def kindle_experiment(data):
    qp.environ['SVMPERF_HOME'] = '../svm_perf_quantification'
    model = OneVsAll(SVMNKLD())
    print(data.training.instances)
    print(data.training.labels.shape)
    model.fit(data.training)
    print("fitted model")
    predicted_labels = model.classify(data.test.instances)
    predicted_prevalence = prevalence_from_labels(predicted_labels)

    true_labels = data.test.labels
    true_prevalence = prevalence_from_labels(true_labels)

    print('true ', true_prevalence)
    print('predicted', predicted_prevalence)


if __name__ == "__main__":
    TRAIN_PATH = "../../data/train_test_data/train_balanced.dat"
    TEST_PATH = "../../data/train_test_data/test.dat"
    data = qp.datasets.fetch_twitter('hcr')
    data = qp.data.Dataset.load(TRAIN_PATH, TEST_PATH, qp.data.reader.from_sparse)
    train_model(data)
    predict(data)
    # kindle_experiment(data)