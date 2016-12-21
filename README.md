# Optimizing functions with computational graphs

A computational graph is simply a representation of an arithmetic expression
using a graph (or in our program, specifically a tree). For example: `x + 2` becomes
```
  +
 / \
x   2
```

`3 (x + 2)` becomes

```
  *
 / \
3   +
   / \
  x   2
```

Why represent expressions this way? By traversing from root to leaves, we can recursively handle each operation in isolation. For example, we can evaluate each node by recursing down its branches and combining the results according to some rule associated with the value or operation at the node -- let's say the operation at the node is `+`; then we add the values returned from recursion down the branches. In general, graphs are a convenient format for dealing with the *structure* of an expression.

In our program, we are not just interested in evaluating expressions, we are interested in *optimizing* them. This means taking certain designated values or **parameters** tweaking them so as to make the value of the expression as large or as small as possible.

How do we do this? The technique that this program uses -- and which is fairly ubiquitous in the world of optimization -- is gradient descent. That means that we calculate the gradient of the function with respect to each of its parameters and then add that gradient to the parameter. A gradient is simply a measure of how much the value of the function changes when the value of a parameter changes. For example,

![](https://github.com/lobachevzky/computational-graph/blob/master/dydx.png)

means that `y` depends on `x` and increasing `x` by one roughly increases `y` by two. Most importantly, it means that increasing `x` also increases `y`. If the result of the expression were negative, it would mean that *decreasing* `x` would increase `y`. Adding gradients to parameters maximizes an expression. Subtracting gradients minimizes it.

Once you have the gradients of an expression, you have a very powerful tool for optimization. But getting those gradients can be tricky. My program does so in a particularly efficient way, using an algorithm that I call *generalized backpropagation*. This algorithm is certainly not original, since there are many programs like Torch and Tensorflow that use it. However, it's not exactly backpropagation because backpropagation applies specifically to the equations of a neural network. *Generalized backpropagation* takes the principles that make backpropagation so powerful for neural networks and generalizes them to *any* expression.

I've put together a little video [here](https://www.youtube.com/watch?v=zhKWBye_RgE&t=117s) that explains this algorithm and demonstrates the improvements in speed that you get using the algorithm over the way you might have been taught to calculate gradients in school. It also has a nice demonstration of the power of using this algorithm with a GPU (in the demo, I use a GeForce GTX TITAN X).

Currently the program implements the following operations for scalars and 2d matrices:
 * negation
 * squaring
 * taking absolute value
 * taking the (elementwise) sign (i.e. +/-)
 * sigmoid
 * tanh
 * multiplication
 * addition
 * subtraction

The library is easily extensible for anyone familiar with Rust and CUDA. This library is obviously not as extensive as Tensorflow or Torch, but it implements the same core capabilities. The goal of this project was not to replace either of those libraries, but to get a better understanding of how they work by reimplementing them myself.

# Performance

This graph compares four implementations of my program: backpropation with naive optimization (explained in detail in the video) both on and off the GPU.
