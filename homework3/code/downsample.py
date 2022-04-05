import numpy as np
from mnist import MNIST
from utils import append

def balance_downsample(images, labels, N):
    n, d = images.shape
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    class_size = int(N/n_classes)

    data = np.c_[images, labels]
    sampled_data = None

    for c in unique_classes:
        class_indices = np.where(labels == c)[0]
        downsampled_indices = np.random.choice(class_indices, class_size, replace=False)
        sampled_data = append(sampled_data, data[downsampled_indices])
        
    np.random.shuffle(sampled_data)
    sampled_images, sampled_labels = sampled_data[:,:-1], np.squeeze(sampled_data[:,-1:])

    return sampled_images, sampled_labels

if __name__ == "__main__":
    mndata = MNIST('./python-mnist/data/')
    images, labels = mndata.load_training()
    images, labels = np.array(images), np.array(labels)

    sampled_images, sampled_labels = balance_downsample(images, labels, 50000)

    np.savetxt("train_downsampled.gz", np.c_[sampled_images, sampled_labels], delimiter=",")