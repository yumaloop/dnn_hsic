# dnn_hsic

### Description

Measuring HSIC(Hilbert Space Independence Critation) between intermediate layers in Deep Neural Network models.

### Experimental Results

The aim of this experiment is to measure the HSICs of each layer in the pre-trained MNIST classifier (DNN model). 
The model classifies the MNIST test data (10,000 images, class pairs) and the state of each layer is taken.

![fig1](https://user-images.githubusercontent.com/31915487/61437436-0b2a4500-a978-11e9-8f0f-52a9844a1559.png)

![fig2](https://user-images.githubusercontent.com/31915487/61437439-0b2a4500-a978-11e9-93ba-48fd37a7e2af.png)

### References

- "Dimensionality Reduction for Supervised Learning with Reproducing Kernel Hilbert Spaces" (2004)<br>
https://www.di.ens.fr/~fbach/fukumizu04a.pdf

- "Measuring Statistical Dependence with Hilbert-Schmidt Norms", (2005)<br>
http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf

- "Kernel Methods for Deep Learning", (NIPS, 2009)<br>
https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf

- "Deep Neural Networks as Gaussian Process", ICLR2018<br>
https://arxiv.org/pdf/1711.00165.pdf

- "Deep Gaussian Process", (2013)<br>
http://proceedings.mlr.press/v31/damianou13a.pdf
