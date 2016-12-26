[![Build Status](https://travis-ci.org/lobachevzky/pipes.svg?branch=gpu)](https://travis-ci.org/lobachevzky/pipes)

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

![](https://github.com/lobachevzky/computational-graph/blob/master/images/dydx.png)

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

# Usage
The master branch is configured to work on CPU. For GPU, switch to the `gpu` branch.
To test the program on a function, you need to write that function in `main.rs` and run `cargo run`, which compiles and runs the program.

Here's an example program:

```rust
let args = HashMap::new();                          // this hashmap can be populated with constants at runtime
let x = Function::param("x", Constant::Scalar(1.)); // a tunable `parameter` Function initialized to 1.
let a = Function::constant(Constant::Scalar(3.));   // a constant scalar Function with value 3.
let f = sq(&(x + a));                               // the function to minimize: (x + a)^2
f.minimize(&args, 0.01, 1000);                      // minimize the function with learning rate of 0.01 and 1000 iterations.
```

An API is pending. For now, check `function/constructors.rs` and `constant/constructors.rs` for different ways to create functions and constants (scalars/matrices).

As for arithmetic operations, most can either take a `Function` type or an `&Function` type (a reference to a `Function`). In general, it is always safe to provide a reference in place of a `Function` since the borrow checker will sometimes complain otherwise.

# Performance

In the next two graphs we compare performance on an LSTM optimizing a randomly generated dataset that takes a sequence of two matrices as input. The first graph compares a GPU with a CPU as the number of tunable parameters increases:

![](https://github.com/lobachevzky/computational-graph/blob/master/images/CPU%20vs%20GPU.png)

The second graph compares the backpropagation algorithm with the naive version of gradient calculation (on the CPU).

![](https://github.com/lobachevzky/computational-graph/blob/master/images/backprop%20vs%20naive.png)

For some reason, the naive algorithm is intractably slow on the GPU and comparisons with it have therefore been ommitted.

Besides the backpropation algorithm itself, a second major optimization used by the algorithm is an object pool for matrices. When a matrix goes out of scope, instead of being deallocated, the program moves the matrix into a global object pool. When the program needs a new matrix, instead of simply allocating one, the program first checks the object pool to see if an unused matrix is available. This the matrices that the backpropation algorithm handles maintain a fixed size, the program only allocates matrices on the first iteration, not subsequently.

The following graph depicts the difference in seconds of optimizing the multiplication of two 2 x 2 matrices with 1000 iterations:

![](https://github.com/lobachevzky/computational-graph/blob/master/images/optimization%20comparison.png)
