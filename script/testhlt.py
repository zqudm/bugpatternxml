from pecos.utils import smat_util
import time
from pecos.xmc.base import HierarchicalKMeans
import scipy.sparse as smat
import numpy as np
from pecos.xmc.base import HierarchicalKMeans
import scipy.sparse as smat
import numpy as np

from scipy.sparse import linalg
from pecos.core import clib as pecos_clib




DATASET = "wiki10-31k"
X = smat_util.load_matrix(f"xmc-base/{DATASET}/tfidf-attnxml/X.trn.npz").astype(np.float32)
Y = smat_util.load_matrix(f"xmc-base/{DATASET}/Y.trn.npz").astype(np.float32)
YT_csr = Y.T.tocsr()
X_csr = X.tocsr()


num_splits = 4
cluster_chain = HierarchicalKMeans.gen(
    X_csr,
    min_codes=num_splits,
    nr_splits=num_splits,
    max_leaf_size=np.ceil(X_csr.shape[0]/num_splits))

print(f"{len(cluster_chain)} layers in the trained hierarchical clusters with C[d] as:")
for d, C in enumerate(cluster_chain):
    print(f"cluster_chain[{d}] is a {C.getformat()} matrix of shape {C.shape}.")



cluster_chain = HierarchicalKMeans.gen(X_csr, nr_splits=8)

print(f"{len(cluster_chain)} layers in the trained hierarchical clusters with C[d] as:")
for d, C in enumerate(cluster_chain):
    print(f"cluster_chain[{d}] is a {C.getformat()} matrix of shape {C.shape}.")



current_cluster = cluster_chain[-1]
for i in range(len(cluster_chain) - 2, -1, -1):
    print(f"{current_cluster.getnnz(0)[0]} instances belong to the first cluster in the layer-{i + 1}.")
    current_cluster = pecos_clib.sparse_matmul(current_cluster, cluster_chain[i])


inst_idx = 10

current_cluster = cluster_chain[-1]
for i in range(len(cluster_chain) - 2, -1, -1):
    print(f"The {inst_idx}-th instance belongs to the cluster-{current_cluster.tocsr().indices[inst_idx]} in the layer-{i + 1}.")
    current_cluster = pecos_clib.sparse_matmul(current_cluster, cluster_chain[i])
