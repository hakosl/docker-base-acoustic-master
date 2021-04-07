import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, mean_squared_error, plot_precision_recall_curve, plot_roc_curve
from visualization.hinton import hinton
from models.sequential_network import SequentialNetwork
from utils.calculate_explicitness import calculate_explicitness
from utils.modularity import compute_deviations, compute_mutual_infos

class NTrainIter():
    def __init__(self, x, y, n, n_folds):
        self.xs = x.shape
        self.ys = y.shape
        self.n = n
        self.n_folds = n_folds
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if i == n_folds:
            raise StopIteration
        else:
            tr_idx = arrange(i * n, (i + 1) * n)
            test_idx = arrange(0, i * n) + arrange((i + 1) * n, self.xs[0])
            if (i + 1) * n > x.shape[0]:
                raise StopIteration
            i += 1
            return tr_idx, test_idx


def label_efficiency(model, dataloader_train, dataloader_val, dataloader_test, n = [60, 240]):
    device = next(model.parameters()).device
    latent_mus, latent_logvars, labels, sample_indexes = get_representation(model, dataloader_train, device)
    latent_mus_v, latent_logvars_v, labels_v, sample_indexes_v = get_representation(model, dataloader_val, device)
    latent_mus_t, latent_logvars_t, labels_t, sample_indexes_t = get_representation(model, dataloader_test, device)
    accs = []
    for i in n:
        clf = GridSearchCV(RandomForestClassifier(n_estimators = 100, random_state=0), {"max_depth": [3, 5, 10, 15, 20]}, "accuracy")

        clf.fit(latent_mus[:i * 10], sample_indexes[:i * 10])
        acc = clf.score(latent_mus_t, sample_indexes_t)
        accs.append(acc)
    
    return n, accs



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

def compute_explicitness(accuracy, n_cat):
    return 1.0 - n_cat * accuracy
    
def compute_compactness(r):
    p = r / np.sum(r, axis=1)[:, np.newaxis]
    c = 1 + np.sum(p * (np.log(p) / np.log(p.shape[1])), axis=1)

    

    return c

def compute_modularity(r):
    p = r / np.sum(r, axis=0)[np.newaxis, :]
    d = 1 + np.sum(p * (np.log(p) / np.log(p.shape[0])), axis=0)

    pm = np.sum(r, axis=0) / np.sum(r)
    modularity = np.dot(d, pm)

    return modularity, d

def one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1

    return b
def compute_mean_auc(model, dataloader):
    device = next(model.parameters()).device
    latent_mus, latent_logvars, labels, sample_indexes = get_representation(model, dataloader, device)
    mean_auc, all_aucs, all_aucs_factors, all_aucs_factor_vals = calculate_explicitness(latent_mus, one_hot(sample_indexes))
    print(all_aucs)
    return np.mean(mean_auc)


def compute_DCI(model, dataloader_train, dataloader_val, dataloader_test, writer, save_hinton=True):
    device = next(model.parameters()).device
    label_names = ["background", "bottom", "other", "sandeel"]
    
    latent_mus, latent_logvars, labels, sample_indexes = get_representation(model, dataloader_train, device)
    latent_mus_v, latent_logvars_v, labels_v, sample_indexes_v = get_representation(model, dataloader_val, device)
    latent_mus_t, latent_logvars_t, labels_t, sample_indexes_t = get_representation(model, dataloader_test, device)

    mean_auc, all_aucs, all_aucs_factors, all_aucs_factor_vals = calculate_explicitness(latent_mus_v, one_hot(sample_indexes_v))
    
    mi = compute_mutual_infos(latent_mus_v, one_hot(sample_indexes_v))
    dev, thet = compute_deviations(mi, label_names)

    
    n_classes = sample_indexes.max() + 1
    predictions = []
    feature_importance = []
    mses = []
    mses_t = []

    

    fig, axs = plt.subplots(1, n_classes, figsize=(12, 4))
    for i in range(n_classes):
        trees_clf = RandomForestClassifier(n_estimators = 250, max_depth=10, random_state=0)
        trees_clf.fit(latent_mus, one_hot(sample_indexes)[:, i])
        
        pred = trees_clf.predict_proba(latent_mus)
        pred_v = trees_clf.predict_proba(latent_mus_v)
        pred_t = trees_clf.predict_proba(latent_mus_t)
        
        mse_v = mean_squared_error(one_hot(sample_indexes_v)[:, i], pred_v[:, 1])
        mses.append(mse_v)

        mse_t = mean_squared_error(one_hot(sample_indexes_t)[:, i], pred_t[:, 1])
        mses_t.append(mse_t)

        feature_importance.append(np.abs(trees_clf.feature_importances_))

        plot_roc_curve(trees_clf, latent_mus_v, one_hot(sample_indexes_v)[:, i], name=label_names[i], ax=axs[i])

    
    writer.add_figure("DCI/ROC", fig)



    feature_importance = np.vstack(feature_importance)
    mse = np.mean(mses)
    mse_t = np.mean(mses_t)

    explicitness = compute_explicitness(mse, n_classes)
    modularity, individual_modularity = compute_modularity(feature_importance)
    compactness = compute_compactness(feature_importance)
    if save_hinton:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        hinton(feature_importance.T, ", ".join(label_names), "$\mathbf{z}$", ax=ax, fontsize=18)
        writer.add_figure("DCI/Hinton", fig)
    
    writer.add_scalar("DCI/modularity", modularity)
    writer.add_scalar("DCI/explicitness", explicitness)
    writer.add_scalar("DCI/information", mse)

    return modularity, explicitness, mse, individual_modularity, compactness




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



