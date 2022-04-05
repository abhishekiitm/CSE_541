from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from utils import load_train_downsampled, preview


def plot_model_selection(d_vals, aic_vals, test_accuracy, variance_expl):
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ln1 = ax.plot(d_vals, aic_vals, label = "AIC value", color="tab:red")
    ax.set_ylabel("AIC")
    ax.set_xlabel("d")

    ax2 = ax.twinx()
    ln2 = ax2.plot(d_vals, test_accuracy, label = "test set accurac", color="tab:blue")
    ln3 = ax2.plot(d_vals, variance_expl, label = "variance explained", color="tab:orange")
    ax2.set_ylabel("percentage")

    lines = ln1+ln2+ln3
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")

    plt.show()

train_images, train_labels = load_train_downsampled("train_downsampled.csv")

mndata = MNIST('./python-mnist/data/')
test_images, test_labels = mndata.load_testing()
test_images, test_labels = np.array(test_images), np.array(test_labels)

# preview(np.array(images), labels, 1)

count_labels = Counter(train_labels)

d_vals = [16, 24, 32, 48, 64, 80, 96, 128]
aic_vals = [0]*len(d_vals)
test_accuracy = [0]*len(d_vals)
variance_expl = [0]*len(d_vals)

for i, d in enumerate(d_vals):
    pca = PCA(n_components=d, svd_solver='full', whiten=True).fit(train_images)

    X_train_pca = pca.transform(train_images)
    variance_expl[i] = sum(pca.explained_variance_ratio_)

    clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log', max_iter=1000, tol=1e-3))
    clf.fit(X_train_pca, train_labels)
    pred = clf.predict_proba(X_train_pca)

    aic_vals[i] = 2*log_loss(train_labels, pred)*X_train_pca.shape[0] + 2*d

    X_test_pca = pca.transform(test_images)
    test_accuracy[i] = clf.score(X_test_pca, test_labels)
    print(d, aic_vals[i], test_accuracy[i])

plot_model_selection(d_vals, aic_vals, test_accuracy, variance_expl)