# dnn_hsic

### Description

Measuring HSIC(Hilbert Space Independence Critation) between intermediate layers in Deep Neural Network models.

### Experimental Results

The experimental content is to measure the HSICs of each layer in the pre-trained MNIST classifier (DNN model). 
The model classifies the MNIST test data (10,000 images-class samples) and the state of each layer is taken.

![fig1](https://user-images.githubusercontent.com/31915487/61437436-0b2a4500-a978-11e9-8f0f-52a9844a1559.png)

![fig2](https://user-images.githubusercontent.com/31915487/61437439-0b2a4500-a978-11e9-93ba-48fd37a7e2af.png)

|       | input             | lay1                | act1              | lay2                | act2               | lay3                | act3              | lay4             | act4              | lay5              | act5                  |
| ----- | ----------------- | ------------------- | ----------------- | ------------------- | ------------------ | ------------------- | ----------------- | ---------------- | ----------------- | ----------------- | --------------------- |
| input | 2.95E-06          | 0.0001690372894     | 1.45E-05          | 0.000141198873      | 0.0000469996998    | 0.0001351601098     | 0.00008241862139  | 0.0001126127641  | 0.00002569158032  | 0.000009351987592 | 0.0000000004710106661 |
| lay1  | 0.000263440744    | 0.01245696657       | 0.001357019575    | 0.01041677749       | 0.004523675079     | 0.009755719057      | 0.007451573117    | 0.00940600974    | 0.002617222478    | 0.0009929085876   | 0.00000005291088778   |
| act1  |                   |                     |                   |                     |                    |                     |                   |                  |                   |                   |                       |
| lay2  |                   |                     |                   |                     |                    |                     |                   |                  |                   |                   |                       |
| Act2  |                   |                     |                   |                     |                    |                     |                   |                  |                   |                   |                       |
| lay3  |                   |                     |                   |                     |                    |                     |                   |                  |                   |                   |                       |
| act3  |                   |                     |                   |                     |                    |                     |                   |                  |                   |                   |                       |
| lay4  |                   |                     |                   |                     |                    |                     |                   |                  |                   |                   |                       |
| act4  |                   |                     |                   |                     |                    |                     |                   |                  |                   |                   |                       |
| lay5  |                   |                     |                   |                     |                    |                     |                   |                  |                   |                   |                       |
| act5  | 0.000001583722253 | 0.00000003030216742 | 0.000001212205375 | 0.00000003181505095 | 0.0000008012884461 | 0.00000006116697143 | 0.000002186409585 | 0.00000145169179 | 0.000006354812953 | 0.000006504984405 | 0.000000002807068279  |

### References

- "Dimensionality Reduction for Supervised Learning with Reproducing Kernel Hilbert Spaces" (JMLR 2004)<br>
https://www.di.ens.fr/~fbach/fukumizu04a.pdf

- "Measuring Statistical Dependence with Hilbert-Schmidt Norms", (ALT 2005)<br>
http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf

- "Kernel Methods for Deep Learning", (NIPS 2009)<br>
https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf

- "Deep Gaussian Process", (AISTATS 2013)<br>
http://proceedings.mlr.press/v31/damianou13a.pdf

- "Deep Neural Networks as Gaussian Process", (ICLR 2018)<br>
https://arxiv.org/pdf/1711.00165.pdf

- "Stronger generalization bounds for deep nets via a compression approach", (ICML 2018)<br>
http://proceedings.mlr.press/v80/arora18b/arora18b.pdf
