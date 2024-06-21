# Faber-Shauder-Function-Approximation
## Inrto
Throughout this repo, we are interested in approximating functions $𝑓 : [0, 1] → \mathbb{R}$ by
ReLU networks. For given function 𝑓 and its approximation ̃︀f' we will use uniform maximum
error as our quality metric:
$$||f  -f'||_C = max_x |f(x) - f'(x)|$$
We will build an approximating neural network using the Faber-Schauder system.

## Faber-Shauder system.
Faber-Shauder system is a basis in space $C(0, 1)$. Moreover, the coefficients of the decomposition:
$$f(x) = \sum_{n = 0}^{\infty} \phi_n(x) A_n(f) $$

Can be calculated by the formulas:
$$A_0(f) = f(0), \quad A_1(f) = f(1) - f(0),$$
$$A_n(f) = A_{k, i}(f) = f(\frac{2i - 1}{2^{k+1}}) - \frac{1}{2}(f(\frac{i - 1}{2^k}) + f(\frac{i}{2^k})), \quad n = 2^k + i, k \in \mathbb{Z}_+, i = 1, 2, \ldots, 2^k$$

Faber-Shauder system is a basis in space $C(0, 1)$.
 
## The main idea

We will recursively build basis functions by ReLU network up to some step and after it sum the results to get an approximation. The number of basis functions depends on the degree of approximation. For more technical details, you can check the article In this repo you can find files with implementation, graphs and auxiliary functions for testing.

`models_approx.py` - main file with implementation of nearal network. For now it has three functions:

1. `square_approximation` - approximation of $f(x) = x^2$
2. `cube_approximation` - approximation of $f(x) = x^3$
3. `exponent_approximation` - approximation of $f(x) = e^x$
   
All of these networks take two arguments for initiation:
1. `k_iter` - number of basis functions. More basis functions - better score.
2. `input_size` - size of input vector

`test.ipynb` - file with some tests and simple examples on how to work with these NN

`visualization.py` - file with auxiliary functions for metrics and graphs. You can see how it works in file `test.ipynb`
