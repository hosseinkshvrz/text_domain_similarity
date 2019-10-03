# Text Domain Similarity
This project includes the implementation of text domain similarity measure in Persian which is described in paper "A Deep Learning-Based Approach for Measuring the Domain Similarity of Persian Texts."

The preprint version of the paper is available at: 
https://arxiv.org/abs/1909.09690

## Usage
The data described in the paper is excluded. You can provide your own data and run the project with it. You just need to specify your data path and file names.

The sequence of execution is as follows: 

1. `preproc_embedding` module conducts preprocessing phase and create word embedding vectors.
2. `ad_pair_maker` generates paired advertisement texts and scores them based on the specified rule.
3. `pair_shuffler` shuffles the prepared dataset, so that the randomness is held.
4. `data_splitter` converts the data in the text file to numpy files each of which contains 1 million paired texts.
5. `cnn_tds`, `lstm_tds` and `w2v_mean_tds` are deep modules which carry out the training, validation and testing phase by the use of `data_loader` and `embedding_loader`.
