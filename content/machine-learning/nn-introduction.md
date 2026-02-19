---
title: Introduction to Neural Networks
---

$$
    \newcommand{\loss}{\mathcal{L}}
    \newcommand{\Dtrain}{\mathcal{D}\_{\text{train}}}
    \newcommand{\Dval}{\mathcal{D}\_{\text{val}}}
    \newcommand{\din}{d\_{\text{in}}}
    \newcommand{\dout}{d\_{\text{out}}}
    \newcommand{\one}{\mathbf{1}}
    \newcommand{\pdv}[2]{\frac{\partial #1}{\partial #2}}
    \newcommand{\R}{\mathbb{R}}
    \newcommand{\E}{\mathbb{E}}
    \newcommand{\Var}{\mathrm{Var}}
    \newcommand{\Cov}{\mathrm{Cov}}
    \newcommand{\Prob}{\mathbb{P}}
$$

## Loss function

We can describe a Machine Learning Model, as some highly convoluted mathematical function $f(x)$. Loss function measures how bad are the model's predictions compared to expected labels given the input. Formally, given a target $y$ and a prediction under model parameters $\theta$ as $\hat{y} = f_\theta(x)$, loss function would be describing similarity between them, as a number $\loss(y, \hat{y})$, where lower is better. In training we want to find optimal model parameters $\theta^*$ that minimize the empirical risk (average loss on data):

$$
    \theta^* = \arg min _\theta \left[ \frac{1}{n} \sum_i \loss \Big( y_i, f_{\theta}(x_i) \Big) \right].
$$

## Backpropagation

Backpropagation is the chain rule applied to the computational graph to get gradients of a scalar loss $\loss$ with respect to all model parameters $\theta$. Suppose that there is a node in graph that computes $Y = \varphi_{\theta}(X)$. Output is dependent both on input $X$ and layer parameters $\theta$. Suppose that we know that the loss function has a gradient over the output of the layer described as $\pdv{\loss}{Y}$, and we want to find both $\pdv{\loss}{X}$ and $\pdv{\loss}{\theta}$. Then using the chain rule we can evaluate:

$$
    \begin{split}
        \pdv{\loss}{X} &= \pdv{\loss}{Y} \pdv{Y}{X}, \\
        \pdv{\loss}{\theta} &= \pdv{\loss}{Y} \pdv{Y}{\theta}. \\
    \end{split}
$$

The matrix $\pdv{Y}{X}$ is sometimes referred to as Jacobian with entries $J_{ij} = \pdv{y_i}{x_i}$.

## Linear Layers

Linear Layer applies an affine map to each input vector:

$$
    Y = XW^T +B \iff y_{ij} = x_{ik}w_{kj}^T + \one_i b_j.
$$

The size of variables is as follows ($N$ is the batch size): $X$ - $(N, \din)$, $W$ - $(\dout, \din)$, and $B$ - $(\dout)$. Suppose that the layer received an upstream gradient equal to $\pdv{\loss}{y}$. Now we need to calculate partials w.t.r. to the layer's weights and activation ($\pdv{\loss}{W}, \pdv{\loss}{B}, \pdv{\loss}{X}$). Using chain rule and tensor derivatives:

$$
    \begin{split}
        \pdv{\loss}{w_{pq}} &= \pdv{\loss}{y_{ij}} \pdv{y_{ij}}{w_{pq}} = \pdv{\loss}{y_{ij}} \pdv{w_{kj}^T}{w_{pq}} x_{ik} \\
        &= \pdv{\loss}{y_{ij}} \delta_{jp}\delta_{kq} x_{ik} = \pdv{\loss}{y_{ip}} x_{iq} \\
        & \implies \pdv{\loss}{W} = \left[ \pdv{\loss}{Y} \right]^T X.
    \end{split}
$$

Doing the same for $B$ yields:

$$
    \begin{split}
        \pdv{\loss}{b_{p}} &= \pdv{\loss}{y_{ij}} \pdv{y_{ij}}{b_{p}} = \pdv{\loss}{y_{ip}} \one_i\\
        & \implies \left[ \pdv{\loss}{B} \right]_p = \sum_i \left[ \pdv{\loss}{Y} \right]_{ip}
    \end{split}
$$

And for $X$:

$$
    \begin{split}
        \pdv{\loss}{x_{pq}} &= \pdv{\loss}{y_{ij}} \pdv{y_{ij}}{x_{pq}} = \pdv{\loss}{y_{pj}} w_{jq} \\
        & \implies \pdv{\loss}{X} = \pdv{\loss}{Y} W.
    \end{split}
$$

## Activation Functions

Between the layers, we introduce activation functions to add non-linearity to the models. For each activation function we need to calculate both forward pass and derivative, so the flow of gradients through the model is undisturbed. Known activation functions are:

### ReLU

![RELU left](machine-learning/res/nn-introduction/relu.png)

For input $X$ and output $Y$ can be defined as $Y = \max(X, 0)$. Outflow of gradient will be the incoming gradient ($\pdv{\loss}{Y}$) into the layer times the derivative:

$$
    \pdv{\loss}{X} = \pdv{\loss}{Y} \pdv{Y}{X} = \pdv{\loss}{Y} \text{sgn}(\max(X,0)).
$$

<div class="clear"></div>

### Sigmoid

![Sigmoid left](machine-learning/res/nn-introduction/sigmoid.png)

This simple activation functions maps any real value $\sigma: \R \rightarrow (0, 1)$ to the valid probability range. Forward pass and backpropagation are defined as:

$$
    \begin{split}
        y &= \sigma(x) = \frac{1}{1+e^x}, \\
        \pdv{y}{x} &= \sigma(x) \Big(1 - \sigma(x) \Big).
    \end{split}
$$

This is often used only for binary classification where value close to zero describes prediction of one class, and values close to one represent the other class.

<div class="clear"></div>

### Softmax

Softmax is a multidimensional sigmoid. Instead of normalizing one value to interval $(0, 1)$ it normalizes vector of values so it represents a valid probability distribution. To satisfy this claim, each element needs to be greater that 0 and the sum of all elements must be equal to $1$. Softmax is can also be denoted as $Y = \sigma_s(X)$. Calculating the gradients of $x,y$ give:

$$
    \begin{split}
        y_{i} &= e^{x_{i}}Z^{-1}, \; Z = \sum_j e^{x_{j}} \\
        \pdv{y_i}{x_p} &= y_i\delta_{ip} - y_i \cdot \frac{e^{x_p}}{Z} = y_i \left( \delta_{ip} - y_p \right).
    \end{split}
$$

Building the Jacobian yields $J_{ij} = \pdv{y_i}{x_j} \iff J = \text{diag}(Y) - YY^T$. Next we can build partial for entire vector $X$:

$$
    \label{eq:softmax}
    \begin{split}
        \left[ \pdv{\loss}{X} \right]_p &= \pdv{\loss}{y_i}\pdv{y_i}{x_p} = \pdv{\loss}{y_i} y_i \left( \delta_{ip} - y_p \right) \\
        &=y_p \left(\pdv{\loss}{y_p} - \pdv{\loss}{y_i}y_i \right)
    \end{split}
$$

This already is a nice form to implement because the element-wise sum of can be nicely casted along batches. Because $\sigma_s(x) = \sigma_s(x+c)$, we can subtract current maximum value $x$ when calculating $y = \sigma_s(x - \max(x))$ to increase numerical stability. Max operation needs to be done per object in the batch, not over the entire input matrix.

### Hyperbolic Tangent

![Tanh left](machine-learning/res/nn-introduction/tanh.png)

Inverse hyperbolic tangent activation can be described as $Y = \tanh^{-1}(X)$, with outputs:

$$
    \begin{split}
        y_i &= \frac{e^{x_i} - e^{-x_i}}{e^{x_i} + e^{-x_i}}, \\
        \pdv{y_i}{x_p} &= (1 - y^2_p) \delta_{ip}.
    \end{split}
$$

Plugging it in similar equation as in softmax function, gives:

$$
    \left[ \pdv{\loss}{X} \right]_p = \pdv{\loss}{y_i} \delta_{ip} \left( 1 - y^2_p \right) = \pdv{\loss}{y_p} \left( 1 - y^2_p \right).
$$

The backpropagation step is reduced to simple element-wise operation on gradient and output of activation function.

## Dropout

We can treat dropout as a standalone layer in the neural network. It applies a binary mask to the input tensor with a specified probability of \emph{keeping} single value equal to $p$. The mask can be interpreted as a matrix $D_{ij} \sim \text{Bernoulli}(p)$. The output $Y$ and the gradient of loss dependent on $X$ are:

$$
    \begin{split}
        Y_{ij} &= \frac{D_{ij}}{p} X_{ij}, \\
        \pdv{\loss}{x_{ij}} &=  \pdv{\loss}{y_{ij}} \pdv{y_{ij}}{x_{ij}} = \pdv{\loss}{y_{ij}} \cdot \frac{D_{ij}}{p}.
    \end{split}
$$

We divide the output by $p$ to keep the expected activation equal. If we didn't the network could fit to receiving lower values during training, and when the dropout is turned off (inferring the model) it would cause errors.

## Optimizers

Optimizers help _apply_ the gradients to the networks weights in order to find the global minimum. Technically we want to minimize the loss $\loss$, so we tweak the model weights $\theta$ in the direction of negative gradient $-\pdv{\loss}{\theta}$. In practice its impossible to make an update using entire training data, so we use batches. If we calculate cumulative gradient over a batch $g_t$ we arrive at SGD (Stochastic Gradient Descent).

### Vanilla SGD

Given model parameters at time $t$ are $\theta_t$, then the update is described as:

$$
    \theta_{t+1} = \theta_t - \eta g_t,
$$

where $\eta$ represents the learning rate.

### Momentum based

Momentum based optimizers introduce new parameter $v_t$ which represents \emph{stacked} velocity along gradients. This aggregation should makes the convergence faster, because the optimizer faster moves along areas with flat gradients.

$$
    \begin{split}
        v_t &= \mu v_{t-1} - \eta g_t, \\
        \theta_{t+1} &= \theta_t + v_t,
    \end{split}
$$

where $\mu$ is the decay speed (or the aggregation) of the velocity.

### Adaptive methods

Two known methods are AdaGrad and RMSProd. The latter introduces exponential moving averages of squared gradients which reduces oscillations.

$$
    \begin{split}
        s_t &= \rho s_{t-1} + (1 - \rho) g_t^2, \\
        \theta_{t+1} &= \theta_t - \eta * \frac{g_t}{\sqrt{s_t + \varepsilon}}
    \end{split}
$$

Parameter $s_t$ is a linear interpolation of previous value $s_t$ and new square of gradients $g_t^2$ (power is taken element-wise). Then the gradients are normalized by dividing by the square root of geometric series of previous gradients. $\varepsilon$ guards the division by $0$.

### Adam - mixed method

Method which merges both \textbf{Ada}ptive and \textbf{M}omentum approaches. The step is given by:

$$
    \begin{split}
        m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, \\
        v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \\
        \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \; \; \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \\
        \theta_{t+1} &= \theta_t - \eta * \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \varepsilon}}.
    \end{split}
$$

Here we introduce two LERP-ing terms $\beta_1, \beta_2$ which control the decay of momentum and square of gradients. Because we start with $m_0 = v_0 = 0$, the EMAs (Exponential Moving Averages) are biased toward $0$ during startup. This normalization negates this effect. If we apply weight decay to this equation, we arrive at AdamW optimizer algorithm.
