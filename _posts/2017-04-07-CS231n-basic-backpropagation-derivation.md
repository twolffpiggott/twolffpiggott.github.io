---
layout: post
title: CS231n basic backpropagation derivation
---
This post covers the backpropagation derivation for an affine layer in a basic fully-connected neural network, as part of work on the second assignment for the Winter 2016 iteration of the Stanford class [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/index.html).

Consider the loss function for the net evaluated at the $i$th observation

$$
\begin{align}
\mathcal{L}_i &= - \log\left(p_{y_i}\right),
\end{align}
$$

where $p_{y_i}$ is the vector of normalised probabilities

$$
\begin{align}
p_{y_i} &= \frac{e^{f_{y_i}}}{\sum_{j=1}^K e^{f_j}},
\end{align}
$$

$f_j = \boldsymbol{w_j}^\top \boldsymbol{x_i} + b_j$ is the score of the $i$th observation corresponding to the $j$th class, $\boldsymbol{x_i}$ is the $D$-dimensional column vector corresponding to the $i$th observation, and $\boldsymbol{w_j}$ is the column vector corresponding to the weights for the $j$th class, for $i=1,\cdots,N$ and $j=1,\cdots,K$.

Given the upstream derivative of the loss with respect to the scores

$$
\begin{align}
\frac{\partial\mathcal{L}_i}{\partial f_k} &= p_k - \mathbb{1}\left(y_i=k\right)
\end{align}
$$

we're interested in deriving the derivative of the loss with respect to the weights and biases. Starting with the weights, we can use the chain rule to get to the desired derivative via the derivatives of the loss with respect to the scores 

$$
\begin{align}
\frac{\partial\mathcal{L}_i}{\partial \boldsymbol{w_j}} &= \sum_{k=1}^K \frac{\partial\mathcal{L}_i}{\partial f_k}\times\frac{\partial f_k}{\partial \boldsymbol{w_j}}\\
&= \frac{\partial\mathcal{L}_i}{\partial f_j}\times\frac{\partial f_j}{\partial \boldsymbol{w_j}}\\
&= \frac{\partial\mathcal{L}_i}{\partial f_j}\boldsymbol{x_i},\\
\end{align}
$$

which follows from the derivative $\frac{\partial f_k}{\partial \boldsymbol{w_j}} = \boldsymbol{x_j}\mathbb{1}\left(j=k\right)$. The $j$th score is the only score where the vector $\boldsymbol{w_j}$ arises, and so the derivatives of the other scores with respect to $\boldsymbol{w_j}$ are all zero. The total loss $\mathcal{L}$ is obtained as the summation of the losses across all $N$ observations, and so

$$
\begin{align*}
\frac{\partial\mathcal{L}}{\partial \boldsymbol{w_j}} &= \sum_{i=1}^N \frac{\partial\mathcal{L}_i}{\partial f_j}\boldsymbol{x_i}.
\end{align*}
$$

This matrix of the derivatives of the loss with respect to the weights is simply obtained as the matrix product of the observation matrix and the matrix of the derivatives of the loss with respect to the scores
$$
\begin{align*}
\begin{bmatrix}
    \sum_{i=1}^N \frac{\partial\mathcal{L}_i}{\partial f_1}\boldsymbol{x_i} & \cdots & \sum_{i=1}^N \frac{\partial\mathcal{L}_i}{\partial f_K}\boldsymbol{x_i}\\
\end{bmatrix}
&=
\begin{bmatrix}
    x_{11} & \cdots & x_{N1}\\
    \vdots & \ddots & \vdots\\
    x_{1D} & \cdots & x_{ND}\\
\end{bmatrix}
\times
\begin{bmatrix}
    \frac{\partial\mathcal{L}_1}{\partial f_1} & \cdots & \frac{\partial\mathcal{L}_1}{\partial f_K}\\
    \vdots & \ddots & \vdots\\
    \frac{\partial\mathcal{L}_N}{\partial f_1} & \cdots & \frac{\partial\mathcal{L}_N}{\partial f_K}\\
\end{bmatrix}\\
dW &= X^\top dF
\end{align*}
$$

The derivative of the loss with respect to the biases is obtained similarly. Again using the chain rule,

$$
\begin{align}
\frac{\partial\mathcal{L}_i}{\partial b_j} &= \sum_{k=1}^K \frac{\partial\mathcal{L}_i}{\partial f_k}\times\frac{\partial f_k}{\partial b_j}\\
&= \frac{\partial\mathcal{L}_i}{\partial f_j}\times\frac{\partial f_j}{\partial b_j}\\
&= \frac{\partial\mathcal{L}_i}{\partial f_j}\times 1,\\
\end{align}
$$

which follows from the derivative $\frac{\partial f_k}{\partial \boldsymbol{b_j}} = \mathbb{1}\left(j=k\right)$. The derivative of the loss across all observations is again obtained via summation 

$$
\begin{align*}
\frac{\partial\mathcal{L}}{\partial b_j} &= \sum_{i=1}^N \frac{\partial\mathcal{L}_i}{\partial b_j}\\
&= \sum_{i=1}^N \frac{\partial\mathcal{L}_i}{\partial f_j},
\end{align*}
$$

and therefore 

$$
\begin{align}
db &= \begin{bmatrix}
    \sum_{i=1}^N \frac{\partial\mathcal{L}_i}{\partial f_1} \\
    \vdots\\
    \sum_{i=1}^N \frac{\partial\mathcal{L}_i}{\partial f_K}\\
\end{bmatrix}
=
\begin{bmatrix}
    \sum_{i=1}^N dF_{i1} \\
    \vdots\\
    \sum_{i=1}^N dF_{iK}\\
\end{bmatrix}.\\
\end{align}
$$

A great walkthrough of the implementation is python is available [here](https://cs231n.github.io/neural-networks-case-study/).
