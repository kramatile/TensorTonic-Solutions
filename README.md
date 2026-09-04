# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Yassine Ben Fatah's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/benfatahyassine.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Bernoulli Probability Mass Function & Moments | Compute the Bernoulli probability mass function, expected value, and variance for a valid success probability. | https://www.tensortonic.com/problems/bernoulli-pmf |
| Binomial Probability Mass Function | Compute binomial probability mass and cumulative probabilities from trial count, success probability, and outcome. | https://www.tensortonic.com/problems/binomial-pmf-cdf |
| Bootstrap Mean & Confidence Interval | Estimate a sample mean and confidence interval through reproducible bootstrap resampling of numeric observations. | https://www.tensortonic.com/problems/bootstrap-mean |
| Catalog Coverage | Measure recommendation catalog coverage as the fraction of available items appearing across user recommendation lists. | https://www.tensortonic.com/problems/catalog-coverage |
| Cosine Annealing LR Scheduler | Compute a cosine-annealed learning rate between configured maximum and minimum values across training steps. | https://www.tensortonic.com/problems/cosine-annealing-lr |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Decision Tree Best Split | Find the best decision-tree split by evaluating candidate feature thresholds and selecting the largest impurity reduction. | https://www.tensortonic.com/problems/decision-tree-split |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Calculate Eigenvalues of a Matrix | Calculate the eigenvalues of a square matrix and return them in the format required by the numerical contract. | https://www.tensortonic.com/problems/eigenvalues |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Compute Gini Impurity for a Split | Compute weighted Gini impurity for a candidate decision-tree split from the class labels on both sides. | https://www.tensortonic.com/problems/gini-impurity |
| Gradient Clipping (Global Norm) | Clip a NumPy gradient array by its global L2 norm while preserving direction when scaling is required. | https://www.tensortonic.com/problems/gradient-clipping |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| He Initialization | Scale raw weights into the He uniform range using a bound derived from the layer fan-in. | https://www.tensortonic.com/problems/he-initialization |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Hit Rate at K | Calculate recommendation hit rate at K by checking whether each user's relevant items appear in top-ranked results. | https://www.tensortonic.com/problems/hit-rate-at-k |
| Compute Information Gain for a Split | Compute information gain for a decision-tree split from parent entropy and weighted child entropies. | https://www.tensortonic.com/problems/information-gain |
| Jaccard Similarity | Compute Jaccard similarity between two collections as intersection size divided by union size. | https://www.tensortonic.com/problems/jaccard-similarity |
| K-Means Assignment Step | Assign each sample to its nearest K-means centroid using Euclidean distance and deterministic tie handling. | https://www.tensortonic.com/problems/k-means-assignment |
| K-Means Centroid Update | Update K-means centroids as cluster means while applying the required behavior for empty clusters. | https://www.tensortonic.com/problems/k-means-centroid-update |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete probability distributions with safe zero-probability handling. | https://www.tensortonic.com/problems/kl-divergence |
| KNN Distance + Neighbor Lookup | Find the nearest neighbors of a query point by computing and ordering Euclidean distances to training samples. | https://www.tensortonic.com/problems/knn-distance |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Matrix Factorization SGD Step | Perform one matrix-factorization SGD update for a rated user-item pair with latent factors and regularization. | https://www.tensortonic.com/problems/matrix-factorization-sgd-step |
| Matrix Inverse | Compute a square matrix inverse in NumPy while returning no result for invalid, non-square, or singular inputs. | https://www.tensortonic.com/problems/matrix-inverse |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max Pooling Forward | Apply 2D max pooling to a numeric matrix using a configurable square window and stride. | https://www.tensortonic.com/problems/maxpool-forward |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| NDCG (Normalized Discounted Cumulative Gain) | Calculate normalized discounted cumulative gain at K from ranked relevance scores and their ideal ordering. | https://www.tensortonic.com/problems/ndcg |
| Implement Nesterov Momentum (NAG) | Implement a Nesterov accelerated-gradient update using lookahead momentum and the current gradient. | https://www.tensortonic.com/problems/nesterov-momentum |
| PCA Projection | Project centered observations onto supplied principal components to produce lower-dimensional features. | https://www.tensortonic.com/problems/pca-projection |
| Compute Pearson Correlation Matrix | Compute the Pearson correlation matrix between numeric features using centered covariance and standard deviations. | https://www.tensortonic.com/problems/pearson-correlation |
| Poisson Probability Mass Function & Cumulative Distribution Function | Compute Poisson probability mass and cumulative probabilities for a nonnegative event count and rate. | https://www.tensortonic.com/problems/poisson-pmf-cdf |
| Popularity Ranking | Rank items by interaction popularity with deterministic ordering for equal-frequency items. | https://www.tensortonic.com/problems/popularity-ranking |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Random Forest Majority Vote | Combine multiple decision-tree predictions with majority voting and deterministic handling of tied classes. | https://www.tensortonic.com/problems/random-forest-vote |
| Rating Normalization | Normalize user-item ratings around the required mean while preserving missing-rating markers. | https://www.tensortonic.com/problems/rating-normalization |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Ridge Regression | Fit ridge regression with L2 regularization using the closed-form solution required by the problem. | https://www.tensortonic.com/problems/ridge-regression |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| One-Sample t-Test | Compute a one-sample t-statistic in NumPy using the sample mean, Bessel-corrected deviation, and hypothesized mean. | https://www.tensortonic.com/problems/t-test-one-sample |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Target Encoding | Encode each categorical value with the mean target observed for its category while preserving row order. | https://www.tensortonic.com/problems/target-encoding |
| Top-K Recommendations | Return each user's highest-scoring unseen items with deterministic ranking and a configurable result limit. | https://www.tensortonic.com/problems/top-k-recommendations |
| Warmup + Linear Decay LR Schedule | Compute a learning-rate schedule with linear warmup followed by linear decay across training steps. | https://www.tensortonic.com/problems/warmup-decay-lr |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/benfatahyassine)
<!-- tensortonic:end -->
