---
title: Gradient Boosting
---

Gradient boosting is based on combining many weak learners and deriving the final decision from this combined "strong" learner. Suppose Least-Squares Regression settings, where the goal is to teach the model $F$ how to approximate and predict value using the function $\hat{y} = F(x)$, by minimizing MSE $\frac{1}{n}\sum (y - \hat{y})^2$. Algorithm goes over $M$ iteration. At each iteration the model is in some imperfect state $F_m$. In order to improve this model, we suppose that we should add some new estimator $h_m(x)$, adjust the next predictor to be:

$$
    F_m(x_i) + h_m(x_i) = y_i,
$$

which yields the new predictor to be:

$$
    h_m(x_i) = y_i - F_m(x_i).
$$

Gradient boosting will fit the predictor $h_m$ to the adjusted residual of the form $y_i - F_m(x_i)$.

## Algorithm

## Gradient Tree Boosting
