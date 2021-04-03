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
from sklearn.metrics import plot_confusion_matrix

from models.sequential_network import SequentialNetwork



def get_representation(model, dataloader, device):
    mus = []
    logvars = []
    labels = []
    si = []
    model.eval()
    for (inputs, label, index) in dataloader:

        m, l = model.encoder(inputs.float().to(device))
        mus.append(m.detach().cpu().numpy())
        logvars.append(l.detach().cpu().numpy())

        labels.append(label)
        si.append(index)

    mus = np.stack(mus)
    mus = mus.reshape(-1, mus.shape[-1])
    logvars = np.stack(logvars)
    logvars = logvars.reshape(-1, logvars.shape[-1])
    labels = np.stack(labels)
    si = np.stack(si)
    si = si.reshape(-1)

    return mus, logvars, labels, si
    

    

def validate_clustering(model, clusterer, dataloader_train, dataloader_test, samplers_test, device, capacity, vb, fig_path="output/clustering.png", i=0, n_visualize=250, save_plot=True, writer=None, dataloader=None):
    enc = model.encoder

    
    latent_mus = []
    latent_logvars = []
    sample_indexes = []
    
    latent_mus, latent_logvars, labels, sample_indexes = get_representation(model, dataloader_train, device)
    latent_mus_t, latent_logvars_t, labels_t, sample_indexes_t = get_representation(model, dataloader_test, device)

    # latent_mu, latent_logvar = enc(test_inputs.to(device))
    # latent_mus = latent_mu.data.cpu().numpy()
    # latent_logvars = latent_logvar.data.cpu().numpy()
    # sample_indexes = si_test.data.cpu().numpy()
    
    # latent_mus = latent_mus.reshape(latent_mus.shape[0], -1)
    #me = PCA(n_components=3, random_state = 42).fit_transform(latent_mus)
    clusterer.fit(latent_mus)
    best_labels = clusterer.labels_

    best_r_score = adjusted_rand_score(best_labels, sample_indexes)
    #r_score = adjusted_rand_score(best_labels, sample_indexes)

    pca = PCA(n_components=2, random_state = 42).fit(latent_mus)   
    me = pca.transform(latent_mus)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))

    ax[0].scatter(me[:n_visualize][:, 0], me[:n_visualize][:, 1], c=best_labels[:n_visualize])
    ax[0].set_title("DBSCAN clusters")

    colors = np.array(["r", "g", "b", "tab:orange", "purple", "cyan"])
    for si in np.unique(sample_indexes[:n_visualize]):
        sm = sample_indexes[:n_visualize] == si
        ax[2].scatter(me[:n_visualize][sm, 0], me[:n_visualize][sm, 1], alpha=0.4, c=colors[si], label=str(samplers_test[:n_visualize][si]))
        ax[2].set_title("original labels")

    ax[2].legend()

    X_train, X_val, y_train, y_val, pca_transformed_train, pca_transformed = train_test_split(latent_mus, sample_indexes, me, test_size=0.2)
    X_train, y_train = latent_mus, sample_indexes
    X_val, y_val = latent_mus_t, sample_indexes_t 
    clf = LogisticRegression(random_state=0, multi_class="auto")
    clf = SVC(decision_function_shape='ovo')
    #clf = SequentialNetwork(capacity, si_test.max() + 1, device, verbose=True)

    
    #clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    if model.__class__.__name__ == "AAESS":
        clf_predictions = model.classify(test_inputs).data.cpu().numpy()
        ax[1].scatter(pca_transformed_train[:n_visualize][:, 0], pca_transformed_train[:n_visualize][:, 1], c=colors[clf_predictions[:n_visualize]])
        ax[1].set_title("AAE semi supervised classifications")
        clf_acc = accuracy_score(sample_indexes, clf_predictions)
    else:
        clf.fit(X_train, y_train)

        cfm = plot_confusion_matrix(clf, X_val, y_val, display_labels=["Background", "Seabed", "Other", "Sandeel"], normalize="true")
        writer.add_figure(f"Confusion matrix classifier {clf.__class__.__name__}", cfm.figure_, i)

        
        clf_predictions = clf.predict(X_val)

        X_val_t = pca.transform(X_val) 

        ax[1].scatter(X_val[:n_visualize][:, 0], X_val[:n_visualize][:, 1], c=colors[clf_predictions[:n_visualize]])
        ax[1].set_title("SVM RBF classifications")


        clf_acc = accuracy_score(y_val, clf_predictions)


    
    fig.suptitle(f"cap: {capacity} beta: {vb} r_score: {best_r_score}, classifier accuracy: {clf_acc}")
    writer.add_figure("Clustering fig", fig, i, close=False)
    if save_plot:
        fig.savefig(fig_path)
    plt.close(fig)
    if save_plot:
        print(f"classifier accuracy: {clf_acc}")


    return best_r_score, clusterer, clf, clf_acc



