import csv
import sys
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from sklearn.cluster import SpectralClustering, OPTICS
from sknetwork.clustering import Louvain, get_modularity


np.random.seed(13)
np.set_printoptions(formatter={'float_kind': '{:f}'.format})


def adjacency(matrix, tol=0.001, method='sqeuclidean'):
    assert method in ('sqeuclidean', 'cosine', 'jaccard')

    A = distance.cdist(matrix.toarray(), matrix.toarray(), metric=method)
    if method == 'sqeuclidean':
        A = 1 / (1 + A)
    if method in ['jaccard', 'cosine']:
        A = 1 - A

    A[A <= tol] = 0
    return csr_matrix(A)


def coverage(adjacency, labels):

    # Code courtesy open-source NOCD community
    # detection implementation
    # Author Oleksandr Shchur and Stephan G
    # Journal: Deep Learning on Graphs Workshop, KDD

    u, v = adjacency.nonzero()
    return (
        ((labels[u] * labels[v]).sum(1) > 0).sum() / adjacency.nnz
    )


def clustering_coef(adjacency, labels):

    # Code courtesy open-source NOCD community
    # detection implementation
    # Author Oleksandr Shchur and Stephan G
    # Journal: Deep Learning on Graphs Workshop, KDD

    def clustering_coef_comm(ind, adjacency):
        adj_comm = adjacency[ind][:, ind]
        n = np.int64(ind.sum())
        if n < 3:
            return 0
        possible = (n - 1) * (n - 1) * n / 6
        existing = (adj_comm @ adj_comm @ adj_comm).diagonal().sum() / 6
        return existing / possible

    labels = labels.astype(bool)
    comm_sizes = labels.sum(0)
    clust_coefs = np.array(
        [clustering_coef_comm(labels[:, c], adjacency)
         for c in range(labels.shape[1])]
    )

    return clust_coefs @ comm_sizes / comm_sizes.sum()


def matricize(vector):
    matrix = np.zeros((vector.size, vector.max() + 1))
    matrix[np.arange(vector.size), vector] = 1

    return matrix


# Use pre-calculated reduced intersection matrices
chan = load_npz('./data_test/games_reduced.npz')
chan_disc = load_npz('./data_test/games_reduced_disc.npz')


with open('./data_test/channel_index.txt', 'r') as f:
    chan_index = [line.strip() for line in f]
with open('./data_test/channel_cols.txt', 'r') as f:
    chan_cols = [line.strip() for line in f]

diag_file = open('./outputs_test/DIAGNOSTICS.csv', 'w', newline='')
DIAGNOSTICS = csv.writer(diag_file, delimiter=',', quotechar='"')
DIAGNOSTICS.writerow([
    'Matrix',
    'Method',
    'Distance',
    'Param 1',
    'Param 2',
    'Param 3',
    'Param 4',
    'Modularity',
    'Coverage',
    'Clustering Coefficient',
    '# Clusters'
])


####################################################

SAVE = False

####################################################

# GENERATE ADJACENCY MATRIX

# Exclude language; measure distance as a continuous value
eps_tuning = {}
for eps in np.linspace(0.3, 0.85, 15):
    A_chan_euc = adjacency(chan, eps, method='sqeuclidean')
    size = 0.5 * (A_chan_euc.shape[0] * (A_chan_euc.shape[0] - 1))
    eps_tuning[eps] = A_chan_euc.nnz / size

eps_avg = {k: abs(v - 0.6) for k, v in eps_tuning.items()}
eps = min(eps_avg, key=eps_avg.get)
A_chan_euc = adjacency(chan, eps, method='sqeuclidean')

print(eps)

# Binary category values, include language
eps_tuning = {}
for eps in np.linspace(0.3, 0.85, 15):
    A_chan_disc = adjacency(chan_disc, 0.1, method='jaccard')
    size = 0.5 * (A_chan_disc.shape[0] * (A_chan_disc.shape[0] - 1))
    eps_tuning[eps] = A_chan_disc.nnz / size

eps_avg = {k: abs(v - 0.6) for k, v in eps_tuning.items()}
eps = min(eps_avg, key=eps_avg.get)
A_chan_disc = adjacency(chan_disc, 0.1, method='jaccard')

print(eps)

if SAVE:
    pd.DataFrame(A_chan_euc.todense(), index=chan_index, columns=chan_index).to_csv(
        './outputs/adjacencies/chan_euc.csv')
    pd.DataFrame(A_chan_disc.todense(), index=chan_index, columns=chan_index).to_csv(
        './outputs/adjacencies/chan_jac.csv')
print('done tuning')

####################################################

SPECTRAL, LOUVAIN, PROP = True, True, True

####################################################

if SPECTRAL:

    # SPECTRAL - CHANNELS

    for n in [10]:
        SC = SpectralClustering(
            n_components=n
        ).fit(chan)
        preds = matricize(SC.labels_)
        save_npz(
            f'./outputs_test/labels/Spectral_euc.npz',
            csr_matrix(preds))
        DIAGNOSTICS.writerow([
            'Channels',
            'Spectral',
            f"{SC.get_params()['affinity']}",
            n,
            '',
            '',
            '',
            get_modularity(A_chan_euc, SC.labels_),
            coverage(A_chan_disc, preds),
            clustering_coef(A_chan_disc, preds),
            len(set(SC.labels_))
        ])
        diag_file.flush()
        print(f'Channel SC done for {n} euc')

    for n in [10]:
        SC = SpectralClustering(
            n_components=n,
            affinity='precomputed',
            assign_labels='cluster_qr'
        ).fit_predict(A_chan_disc)
        preds = matricize(SC)
        save_npz(
            f'./outputs_test/labels/Spectral_jac.npz',
            csr_matrix(preds))

        DIAGNOSTICS.writerow([
            'Channels',
            'Spectral',
            'Jaccard',
            n,
            '',
            '',
            '',
            get_modularity(A_chan_disc, SC),
            coverage(A_chan_disc, preds),
            clustering_coef(A_chan_disc, preds),
            len(set(SC))
        ])
        diag_file.flush()
        print(f'Channel SC done for {n} jac')


####################################################


if PROP:

    # PROPAGATION - CHANNELS

    PC = OPTICS().fit_predict(A_chan_euc.toarray())
    preds = matricize(PC)
    save_npz(
        './outputs_test/labels/Prop_euc.npz',
        csr_matrix(preds))

    DIAGNOSTICS.writerow([
        'Channel',
        'Propagation',
        'Euclidean',
        '',
        '',
        '',
        '',
        get_modularity(A_chan_euc, PC),
        coverage(A_chan_disc, preds),
        clustering_coef(A_chan_disc, preds),
        len(set(PC))
    ])
    diag_file.flush()
    print(f'Channel PC done for euc')

    PC = OPTICS().fit_predict(A_chan_disc.toarray())
    preds = matricize(PC)
    save_npz(
        './outputs_test/labels/Prop_jac.npz',
        csr_matrix(preds))

    DIAGNOSTICS.writerow([
        'Channel',
        'Propagation',
        'Jaccard',
        '',
        '',
        '',
        '',
        get_modularity(A_chan_disc, PC),
        coverage(A_chan_disc, preds),
        clustering_coef(A_chan_disc, preds),
        len(set(PC))
    ])
    diag_file.flush()
    print(f'Channel PC done for jac')


####################################################


if LOUVAIN:

    # LOUVAIN - CHANNELS

    LV = Louvain().fit_predict(A_chan_euc)
    preds = matricize(LV)
    save_npz(
        './outputs_test/labels/Louvain_euc.npz',
        csr_matrix(preds))

    DIAGNOSTICS.writerow([
        'Channel',
        'Louvain',
        'Euclidean',
        '',
        '',
        '',
        '',
        get_modularity(A_chan_euc, LV),
        coverage(A_chan_disc, preds),
        clustering_coef(A_chan_disc, preds),
        len(set(LV))
    ])
    diag_file.flush()
    print(f'Channel LV done for euc')

    LV = Louvain().fit_predict(A_chan_disc)
    preds = matricize(LV)
    save_npz(
        './outputs_test/labels/Louvain_jac.npz',
        csr_matrix(preds))

    DIAGNOSTICS.writerow([
        'Channel',
        'Louvain',
        'Jaccard',
        '',
        '',
        '',
        '',
        get_modularity(A_chan_disc, LV),
        coverage(A_chan_disc, preds),
        clustering_coef(A_chan_disc, preds),
        len(set(LV))
    ])
    diag_file.flush()
    print(f'Channel LV done for jac')


####################################################

# NOCD - CHANNELS


# NOCD - USERS


diag_file.close()
