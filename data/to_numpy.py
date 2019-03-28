import os
import arff
import argparse
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

def get_features(path, features_dim):
    file_content = arff.load(open(path, "r"))
    data=np.array(file_content['data'], dtype="float32")
    return data[:, : features_dim]

def get_labels(path, features_dim, labels_dim):
    file_content = arff.load(open(path, "r"))
    data=np.array(file_content['data'], dtype="float32")
    return data[: , features_dim : ]

def set_dims(dataset_path):
    with open(os.path.join(dataset_path, "count.txt"), "r") as f:
        return [int (i) for i in f.read().split("\n") if i != ""]

if __name__ == '__main__':
    # Convert and save arff files to numpy-pickles for faster data I/O.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mirflickr", help="Name of the dataset")
    parser.add_argument("--load_arff_split", default=True, help="Load arff data is splited")

    args = parser.parse_args()
    dataset = args.dataset
    load_arff_split = args.load_arff_split
    features_dim, labels_dim = set_dims("./{}/".format(dataset))

    if load_arff_split == False:
        features, labels = get_features("./{}/{}.arff".format(dataset, dataset), features_dim), get_labels("./{}/{}.arff".format(dataset, dataset), features_dim, labels_dim)
        features.dump("./{}/{}-features.pkl".format(dataset, dataset))
        labels.dump("./{}/{}-labels.pkl".format(dataset, dataset))

        train_features,test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)
        train_features.dump("./{}/{}-train-features.pkl".format(dataset, dataset))
        train_labels.dump("./{}/{}-train-labels.pkl".format(dataset, dataset))
        test_features.dump("./{}/{}-test-features.pkl".format(dataset, dataset))
        test_labels.dump("./{}/{}-test-labels.pkl".format(dataset, dataset))

        print(train_features.shape, train_labels.shape)
    else:
        #train_features, train_labels = get_features("./{}/{}-train.arff".format(dataset, dataset), features_dim), get_labels("./{}/{}-train.arff".format(dataset, dataset), features_dim, labels_dim)
        train_features = np.loadtxt('./mirflickr/x_train.txt', delimiter=',', dtype="float32")
        train_labels = np.loadtxt('./mirflickr/y_train.txt', delimiter=',', dtype="float32")
        test_features = np.loadtxt('./mirflickr/x_test.txt', delimiter=',', dtype="float32")
        test_labels = np.loadtxt('./mirflickr/y_test.txt', delimiter=',', dtype="float32")


        train_features.dump("./{}/{}-train-features.pkl".format(dataset, dataset))
        train_labels.dump("./{}/{}-train-labels.pkl".format(dataset, dataset))
        #test_features, test_labels = get_features("./{}/{}-test.arff".format(dataset, dataset), features_dim), get_labels("./{}/{}-test.arff".format(dataset, dataset), features_dim, labels_dim)
        test_features.dump("./{}/{}-test-features.pkl".format(dataset, dataset))
        test_labels.dump("./{}/{}-test-labels.pkl".format(dataset, dataset))
