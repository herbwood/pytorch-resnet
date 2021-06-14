import pickle
import requests
import tarfile

def unpickle(file):
    with open(file, 'rb') as f:
        pickle_dict = pickle.load(f, encoding='bytes')
    
    return pickle_dict 

def download_cifar10(url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 
                     target_path='cifar-10-python.tar.gz'):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())

    if target_path.endswith('tar.gz'):
        tar = tarfile.open(target_path, 'r:gz')
        tar.extractall()
        tar.close()
