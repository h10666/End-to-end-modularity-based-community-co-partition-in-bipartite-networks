# End-to-end modularity-based community co-partition in bipartite networks.
In this paper, we propose BiCoN, an end-to-end learning framework
for community co-partition in bipartite networks. In view of the
challenge of end-to-end deep community detection in networks
with heterogeneity, we extend the widely-used spectral and spatial
graph convolution operators to bipartite scenarios for node feature
encoding. Then we formulate a novel modularity-based loss func-
tion with collapsed regularizations on two assignment matrices.
End-to-end unsupervised community detection can be achieved by
optimizing the loss function with respect to network weights. Com-
prehensive experiments on various types of datasets demonstrate
the efficacy of the proposed method.

## DataSets 

* Unlabeled Datasets: We use the PubMed dataset established
by Yang et al. to generate our own unlabeled bipartite networks for community co-partition. PubMed is an online database of research articles on life sciences and biomedical topics. 
* Labeled Datasets:The first dataset is Citeseer. It contains 3,312 scientific papers as nodes with 6 class labels, and 4,732 citationsas edges among these papers. The second dataset is Cora. This dataset contains 2,708 machine learning papers with 7 labels, and 5,429 citations.  The third citation dataset PubMed (with labels) 
* Large-scale Datasets：  DBLP dataset is a heterogeneous network with three types of nodes, including author, venue and phrases. OGBN-arXiv dataset is a citation network of all the computer science arXiv papers index by MAG