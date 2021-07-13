### Input
Download edgelists & labels from [here](http://www4.comp.polyu.edu.hk/~jiemshi/datasets.html)
Download mat files from [here](https://entuedu-my.sharepoint.com/:f:/g/personal/yang0461_e_ntu_edu_sg/EtSF-oW24J9NqYtt5prKlm0BhYQaTXm2vIqk4Xvt0j7jUw?e=lvpCJV)


### Output
Default output is a binary array.
If you want to output a CSV file, i.e., each line is an embedding vector, please chang Line 128 in nrp.m as format="csv";.

### Functions
1. nrp.m :the node-reweighted PPR embedding algorithm, input arguments are: graph-data-name, if-it-is-full, if-it-is-directed, dimensionality, epoch-number
2. bksvd.m : the randomized SVD algorithm, input arguments are: the square matrix you wanna decompose, the low-rank, others as default
3. update_dw.m : the algorithm for updating forward/backward weights of directed graphs
4. update_uw.m : the algorithm for updating forward/backward weights of undirected graphs


### How to run
```sh
$ matlab -nodisplay -r "cd('.'); nrp('wiki', 0,  1, 128, 20, 10);exit"
```
Parameters: graph-name, full (1) or partial (0), directed (1) or undirected (0), embedding dimensionality, number of iterations for PPR approximation, number of iterations for learning weights. 
