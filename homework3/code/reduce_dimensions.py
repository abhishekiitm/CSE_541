from sklearn.decomposition import PCA
from utils import load_train_downsampled
import numpy as np
import pickle

if __name__=="__main__":
    train_images, train_labels = load_train_downsampled("train_downsampled.csv")

    d = 48

    pca = PCA(n_components=d, svd_solver='full', whiten=True).fit(train_images)
    X_train_pca = pca.transform(train_images)

    # dump state of pca object for reuse
    with open('pca_48.pkl', 'wb') as output:
        pickle.dump(pca, output, pickle.HIGHEST_PROTOCOL)

    np.savetxt("train_downsampled_pca_48.csv", np.c_[X_train_pca, train_labels], delimiter=",")