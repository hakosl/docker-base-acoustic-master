import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from models.sequential_network import SequentialNetwork

def validate_clustering(model, clusterer, test_inputs, si_test, samplers_test, device, capacity, vb, fig_path="output/clustering.png", n_visualize=250):
    enc = model.encoder

    
    latent_mus = []
    latent_logvars = []
    sample_indexes = []
    
    latent_mu, latent_logvar = enc(test_inputs)
    latent_mus = latent_mu.data.cpu().numpy()
    latent_logvars = latent_logvar.data.cpu().numpy()
    sample_indexes = si_test.data.cpu().numpy()
    
    print(f"latent mu shape {latent_mus.shape}")
    latent_mus = latent_mus.reshape(latent_mus.shape[0], -1)
    #me = PCA(n_components=3, random_state = 42).fit_transform(latent_mus)
    clusterer.fit(latent_mus)
    best_labels = clusterer.labels_

    best_r_score = adjusted_rand_score(best_labels, sample_indexes)

    pca = PCA(n_components=2, random_state = 42).fit(latent_mus)   
    me = pca.transform(latent_mus)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))

    ax[0].scatter(me[:n_visualize][:, 0], me[:n_visualize][:, 1], c=best_labels[:n_visualize])
    ax[0].set_title("DBSCAN clusters")

    colors = ["r", "g", "b", "tab:orange", "purple", "cyan"]
    for si in np.unique(sample_indexes[:n_visualize]):
        sm = sample_indexes[:n_visualize] == si
        ax[2].scatter(me[:n_visualize][sm, 0], me[:n_visualize][sm, 1], alpha=0.4, c=colors[si], label=str(samplers_test[:n_visualize][si]))
        ax[2].set_title("original labels")

    ax[2].legend()

    X_train, X_val, y_train, y_val = train_test_split(latent_mus, sample_indexes, test_size=0.2)

    clf = LogisticRegression(random_state=0, multi_class="auto")
    clf = SVC(decision_function_shape='ovo')
    #clf = SequentialNetwork(capacity, si_test.max() + 1, device, verbose=True)

    
    #clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    clf.fit(X_train, y_train)

    clf_predictions = clf.predict(X_val)

    X_val = pca.transform(X_val) 

    ax[1].scatter(X_val[:n_visualize][:, 0], X_val[:n_visualize][:, 1], c=clf_predictions[:n_visualize])
    ax[1].set_title("logreg classifications")
    clf_acc = accuracy_score(y_val, clf_predictions)

    
    fig.suptitle(f"cap: {capacity} beta: {vb} r_score: {best_r_score}")
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"classifier accuracy: {clf_acc}")


    return best_r_score, clusterer, clf



