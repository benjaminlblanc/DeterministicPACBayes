import numpy as np
import gzip
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from category_encoders.ordinal import OrdinalEncoder

from data.utils import get_validation_set, download, read_idx_file
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages when loading the datasets.

def fetch_SVMGUIDE1(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'svmguide1.data'
    test_path = path / 'svmguide1-test.data'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1.t', test_path)

    X_train, y_train = read_idx_file(train_path, 4, " ")
    X_test, y_test = read_idx_file(train_path, 4, " ")

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    Y = np.hstack((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(np.vstack((X_train, X_test)), Y, stratify=Y, test_size=test_size, random_state=seed)
    
    X_val, y_val = get_validation_set(X_train, y_train, valid_size, seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_CODRNA(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'codrna.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna', data_path)

    X, Y = read_idx_file(data_path, 8)
    Y[Y == 0] = -1
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_val, y_val = get_validation_set(X_train, y_train, valid_size / (1 - test_size), seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_PHISHING(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'phishing.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing', data_path)

    X, Y = read_idx_file(data_path, 68, " ")
    Y[Y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_val, y_val = get_validation_set(X_train, y_train, valid_size / (1 - test_size), seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_ADULT(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'adult.data'
    test_path = path / 'adult.test'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t', test_path)

    X_train, y_train = read_idx_file(train_path, 123)
    X_test, y_test = read_idx_file(test_path, 123)

    X, Y = np.vstack([X_train, X_test]), np.hstack([y_train, y_test])
    Y[Y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)
    X_val, y_val = get_validation_set(X_train, y_train, valid_size / (1 - test_size), seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_HABERMAN(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'haberman.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data', data_path)

    data = np.genfromtxt(data_path, delimiter=',')

    X, Y = (data[:, :-1]).astype(np.float32), (data[:, -1] - 1).astype(int)
    Y[Y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_val, y_val = get_validation_set(X_train, y_train, valid_size / (1 - test_size), seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_MUSHROOMS(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'agaricus-lepiota.data'

    if not data_path.exists():
        path.mkdir(parents=True, exist_ok=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', data_path)

    data = pd.read_csv(data_path, names=np.arange(23))
    encoder = OrdinalEncoder(return_df=False)
    data = encoder.fit_transform(data)
    
    X, Y = (data[:, 1:]).astype(np.float32), (data[:, 0] - 1).astype(int)
    Y[Y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_val, y_val = get_validation_set(X_train, y_train, valid_size / (1 - test_size), seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_TICTACTOE(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'tic-tac-toe.data'

    if not data_path.exists():
        path.mkdir(parents=True, exist_ok=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', data_path)

    data = pd.read_csv(data_path, names=np.arange(10))
    encoder = OrdinalEncoder(return_df=False)
    data = encoder.fit_transform(data)
    
    X, Y = (data[:, :-1]).astype(np.float32), (data[:, -1] - 1).astype(int)
    Y[Y == 0] = -1
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_val, y_val = get_validation_set(X_train, y_train, valid_size / (1 - test_size), seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

# MULTICLASS CLASSIFICATION
def fetch_MNIST(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'mnist.scale.bz2'
    test_path = path / 'mnist.scale.t.bz2'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2', test_path)

    X_train, y_train = read_idx_file(train_path, 784, " ", True)
    X_test, y_test = read_idx_file(test_path, 784, " ", True)
    
    Y = np.hstack((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(np.vstack((X_train, X_test)), Y, stratify=Y, test_size=test_size, random_state=seed)
    
    X_val, y_val = get_validation_set(X_train, y_train, valid_size, seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )


def load_DNN_dataset(dataset_name, path, shuffle=True):
    """
    Download (if necessary) DNN database file and extract it.
    Return the tuple of training and testing dataset.
    """
    import os
    import logging
    import tarfile
    import urllib.request
    import pickle
    import numpy as np
    path = Path(path)
    dataset_name_small = dataset_name.split('_')[0]

    DNN_DIR_PATH = path / f'data/{dataset_name_small}/'
    DNN_FOLDERNAME = path / f'data/{dataset_name_small}/{dataset_name_small}-batches-py'
    DNN_BATCH_SIZE = 10000  # data are split into blocks of 10000 images
    if dataset_name_small == 'CIFAR10':
        DNN_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        DNN_TRAINING_FILENAMES = [os.path.join(DNN_DIR_PATH, DNN_FOLDERNAME, 'data_batch_%d' % i) for i in range(1, 6)]
        DNN_TESTING_FILENAMES = [os.path.join(DNN_DIR_PATH, DNN_FOLDERNAME, 'test_batch')]

        def read_DNN_files(filenames):
            """
            Return the DNN dataset loaded from the bunch of files.

            Keyword arguments:
            filenames -- the list of filenames (strings)
            """
            dataset = []  # dataset to be returned
            for file in filenames:
                with open(file, 'rb') as fo:
                    _dict = pickle.load(fo, encoding='bytes')
                data = _dict[b'data']
                labels = _dict[b'labels']
                assert data[0].size == 3 * 32 * 32

                for k in range(DNN_BATCH_SIZE):
                    image = data[k].reshape(3, 32, 32)
                    image = np.transpose(image, [1, 2, 0])
                    dataset.append([image, labels[k]])
            return dataset
        logging.info("Loading dataset ...")
        # checking if the data is already in the folder
        if not os.path.isdir(os.path.join(DNN_DIR_PATH, DNN_FOLDERNAME)):
            # if not, we download the data
            os.makedirs(DNN_DIR_PATH, exist_ok=True) # create folder for the data
            filename = DNN_DATA_URL.split('/')[-1]
            filepath = os.path.join(DNN_DIR_PATH, filename)
            # try to download the file
            try:
                import sys
                def _progress(cnt,blck_size,total_size):
                    sys.stdout.write('\r>> Downloading file %s (%3.1f%%)' % (filename, 100.0*cnt*blck_size/total_size))
                    sys.stdout.flush()
                logging.info("Downloading file {f}".format(f=DNN_DATA_URL))
                fpath, _ = urllib.request.urlretrieve(DNN_DATA_URL, filepath, reporthook=_progress)
                statinfo = os.stat(fpath)
                size = statinfo.st_size
            except:
                logging.error("Failed to download {f}".format(f=DNN_DATA_URL))
                raise

            print('Succesfully downloaded {f} ({s} bytes)'.format(f=filename,s=size))
            tarfile.open(filepath, 'r:gz').extractall(DNN_DIR_PATH)

        trainingData = read_DNN_files(DNN_TRAINING_FILENAMES)
        testingData = read_DNN_files(DNN_TESTING_FILENAMES)

    if shuffle:
        logging.info("Shuffling data ...")
        import sklearn
        trainingData = sklearn.utils.shuffle(trainingData)
        testingData = sklearn.utils.shuffle(testingData)

    return trainingData, testingData

def create_DNN(dataset_name, path):
    import tensorflow as tf
    import time
    import numpy as np
    import sys
    import os

    import urllib.request
    import tarfile

    # Create model directory if it doesn't exist
    os.makedirs(path / 'models/', exist_ok=True)

    # Download URL and file paths
    url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    download_path = path / 'inception-2015-12-05.tgz'
    model_extract_path = path / 'models/classify_image_graph_def.pb'
    data_extract_path = path / f'data/{dataset_name}/{dataset_name}.npz'

    # Extract the specific file from the tar.gz archive
    # Check if model already exists

    if not os.path.exists(model_extract_path):
        # Download the model
        print("Downloading Inception model...")
        urllib.request.urlretrieve(url, download_path)
        print(f"Downloaded to {download_path}")
        print("Extracting model...")
        with tarfile.open(download_path, 'r:gz') as tar:
            # Find the classify_image_graph_def.pb file in the archive
            for member in tar.getmembers():
                if member.name.endswith('classify_image_graph_def.pb'):
                    # Extract and rename to the desired path
                    member.name = 'classify_image_graph_def.pb'  # Remove any directory structure
                    tar.extract(member, path / 'models/')
                    break
            else:
                print("Warning: classify_image_graph_def.pb not found in archive")

        # Clean up the downloaded tar.gz file
        os.remove(download_path)
        print(f"Model ready at: {model_extract_path}")

    if not os.path.exists(data_extract_path):
        # Loading the dataset to transform
        data_training, data_testing = load_DNN_dataset(dataset_name, path, shuffle=False)

        graph_def = tf.compat.v1.GraphDef()
        with open(path / 'models/classify_image_graph_def.pb', "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='Incv3')

        # number of samples to extract from

        nsamples_training = len(data_training)    # at most 50000
        nsamples_testing  = len(data_testing)     # at most 10000

        nsamples = nsamples_training + nsamples_testing

        X_data = [ data_training[i][0] for i in range(nsamples_training) ] + \
                 [ data_testing[i][0]  for i in range(nsamples_testing)  ]

        y_training = np.array( [ data_training[i][1] for i in range(nsamples_training) ] )
        y_testing  = np.array( [ data_testing[i][1]  for i in range(nsamples_testing)  ] )

        # Running tensorflow session in order to extract features
        def _progress(count, start, time):
            percent = 100.0*(count+1)/nsamples
            sys.stdout.write('\r>> Extracting features %4d/%d  %6.2f%%   \
                              ETA %8.1f seconds' % (count+1, nsamples, percent, (time-start)*(100.0-percent)/percent) )
            sys.stdout.flush()

        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            predictions = np.zeros((nsamples, 1008), dtype='float32')
            representations = np.zeros((nsamples, 2048), dtype='float32')

            print('nsamples = ', nsamples)
            start = time.time()
            for i in range(nsamples):
                [reps, preds] = sess.run([representation_tensor, softmax_tensor], {'DecodeJpeg:0': X_data[i]})
                predictions[i] = np.squeeze(preds)
                representations[i] = np.squeeze(reps)
                if i+1==nsamples or not (i%10): _progress(i, start, time.time())
            print('\nElapsed time %.1f seconds' % (time.time()-start))

        # Create data directory if it doesn't exist
        os.makedirs(path / f'data/{dataset_name}/', exist_ok=True)

        # Finally we can save our work to the file
        np.savez_compressed(data_extract_path,
                            features_training=representations[:nsamples_training],
                            features_testing=representations[-nsamples_testing:],
                            labels_training=y_training,
                            labels_testing=y_testing )

def fetch_DNN(dataset_name, path, valid_size=0.2, test_size=0.2, seed=None):
    path = Path(path).parent.parent
    # If necessary, create the dataset
    create_DNN(dataset_name, path)

    # Load the compressed dataset
    data = np.load(path / f'data/{dataset_name}/{dataset_name}.npz')

    # Extract individual arrays using the keys from savez_compressed
    X_train_full = data['features_training']
    X_test = data['features_testing']
    y_train_full = data['labels_training']
    y_test = data['labels_testing']

    data.close()

    Y = np.hstack((y_train_full, y_test))
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(
        (X_train_full, X_test)), Y, stratify=Y, test_size=test_size, random_state=seed)

    X_val, y_val = get_validation_set(X_train, y_train, valid_size, seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_PENDIGITS(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'pendigits.data'
    test_path = path / 'pendigits.t.data'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t', test_path)

    X_train, y_train = read_idx_file(train_path, 16, " ")
    X_test, y_test = read_idx_file(test_path, 16, " ")
    
    Y = np.hstack((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(np.vstack((X_train, X_test)), Y, stratify=Y, test_size=test_size, random_state=seed)
    
    X_val, y_val = get_validation_set(X_train, y_train, valid_size, seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_PROTEIN(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'protein.bz2'
    test_path = path / 'protein.t.bz2'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.bz2', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.t.bz2', test_path)

    X_train, y_train = read_idx_file(train_path, 357, '  ', True)
    X_test, y_test = read_idx_file(test_path, 357, '  ', True)

    Y = np.hstack((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(np.vstack((X_train, X_test)), Y, stratify=Y, test_size=test_size, random_state=seed)
    
    X_val, y_val = get_validation_set(X_train, y_train, valid_size, seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_SENSORLESS(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'sensorless.data'

    if not data_path.exists():
        path.mkdir(parents=True, exist_ok=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless', data_path)

    X, Y = read_idx_file(data_path, 48)
    Y -= 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_val, y_val = get_validation_set(X_train, y_train, valid_size, seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_SHUTTLE(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'shuttle.data'
    test_path = path / 'shuttle.t.data'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/shuttle.scale', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/shuttle.scale.t', test_path)

    X_train, y_train = read_idx_file(train_path, 9, " ")
    y_train -= 1
    X_test, y_test = read_idx_file(test_path, 9, " ")
    y_test -= 1

    Y = np.hstack((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(np.vstack((X_train, X_test)), Y, stratify=Y, test_size=test_size, random_state=seed)
    
    X_val, y_val = get_validation_set(X_train, y_train, valid_size, seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_FASHION_MNIST(path, valid_size=0.2, test_size=0.2, seed=None):
    """code adapted from https://github.com/StephanLorenzen/MajorityVoteBounds/blob/278a2811774e48093a7593e068e5958832cfa686/mvb/data.py#L143"""
    path = Path(path)
    train_path = path / 'fashion-mnist-train.data.gz'
    train_label_path = path / 'fashion-mnist-train.label.gz'
    test_path = path / 'fashion-mnist-test.data.gz'
    test_label_path = path / 'fashion-mnist-test.label.gz'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-images-idx3-ubyte.gz?raw=true', test_path)
        download('https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-labels-idx1-ubyte.gz?raw=true', test_label_path)
        download('https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-images-idx3-ubyte.gz?raw=true', train_path)
        download('https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-labels-idx1-ubyte.gz?raw=true', train_label_path)

    with gzip.open(train_path) as f:
        f.read(16)
        buf = f.read(28*28*60000)
        X_train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(60000, 28*28)  

    with gzip.open(test_path) as f:
        f.read(16)
        buf = f.read(28*28*10000)
        X_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(10000, 28*28)
    
    with gzip.open(train_label_path) as f:
        f.read(8)
        buf = f.read(60000)
        y_train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) 

    with gzip.open(test_label_path) as f:
        f.read(8)
        buf = f.read(10000)
        y_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    Y = np.hstack((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(np.vstack((X_train, X_test)), Y, stratify=Y, test_size=test_size, random_state=seed)
    
    X_val, y_val = get_validation_set(X_train, y_train, valid_size, seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )
