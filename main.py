#!/usr/bin/env python3

import numpy as np
import numpy.random as r

from scipy.stats import norm

xs  = [0, 1]
mu0 = 0
mu1 = 5
p   = 0.95

def px0(x0):
  return (0 * x0) + (1 / len(xs))

# Draw the initial state by choosing uniformly at random from {0, 1}
def draw_x0(size):
  return r.choice(xs, size=size)

# Draw a random transition from state x_t to x_{t+1}
def draw_transition(xt, p=p):
  # "changes" is a pseudo-variable, if it's 1, then xt will switch state.
  changes = r.choice([0, 1], p=[p, 1 - p], size=len(xt))

  # XOR "xt" with "changes" to get x_{t+1}.
  # 00 -> 0; 01 -> 1; 10 -> 1; 11 -> 0;
  xt_next = xt.astype(int) ^ changes
  return xt_next

# Probability of observation gven state
def py_bar_xt(y, x):
  return norm.pdf(y, loc=mu1*x, scale=1)

##### Simulate Model #####

def simulate_forward(num_obs):
  X = np.zeros(num_obs) # list of states
  Y = np.zeros(num_obs) # list of observations

  X[0] = draw_x0(1) # Draw initial state
  Y[0] = r.normal(loc=X[0] * mu1, scale=1) # sample initial observation

  for i in range(1, num_obs):
    x = np.array([X[i - 1]])  # previous state
    X[i] = draw_transition(x) # random transition
    Y[i] = r.normal(loc=X[i] * mu1, scale=1) # generate observation

  return (X, Y) # set of states + observations.

##### Bootstrap particle filter #####

def normalise(ws):
  return ws / ws.sum()

def bootstrap(ys, num_particles):
  # 1. Initialization
  T       = len(ys)
  # A grid T "tall", and num_particles "wide".
  X       = np.zeros(shape=(T, num_particles))
  # draw from x0 N times.
  X[0]    = draw_x0(num_particles)

  indexes = np.array(range(0, num_particles))

  # weight each particle based on probability of observing y_0
  # then normalise + reselect
  wt_tmp = normalise( py_bar_xt( ys[0], X[0] ) )
  ixs    = r.choice(indexes, p=wt_tmp, size=num_particles)
  X = X[:, ixs]

  for t in range(1, T):
    # 2. Importance sampling step
    X[t]   = draw_transition(X[t - 1])
    wt_tmp = normalise( py_bar_xt( ys[t], X[t] ) )

    # 3. Selection step: pick particles weighted by importance.
    ixs = r.choice(indexes, p=wt_tmp, size=num_particles)
    X = X[:, ixs]

  return X

# Maximum likelihood estimation from bootstrap particles
def bootstrap_ml(y, num_particles):
  X     = bootstrap(y, num_particles)
  x_est = X.sum(axis=1) / num_particles
  x_ml  = np.round(x_est)
  return x_ml


##### Viterbi algorithm #####

def viterbi(y):
  # http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-18.html
  # https://onlinecourses.science.psu.edu/stat857/node/203
  T = len(y)

  G = np.zeros((T, len(xs))) # time goes "down"
  L = np.zeros((T, len(xs))).astype(int) - 1

  xt = np.array(xs)
  G[0] = px0(xt) * py_bar_xt(y[0], xt)

  for t in range(1, T):
    for k in xs:
      g_t_k = np.zeros(len(xs))
      for l in xs:
        ptrans = p if k == l else 1 - p
        g_t_k[l] = G[t - 1, l] * ptrans * py_bar_xt(y[t], k)

      L[t, k] = np.argmax(g_t_k)
      G[t, k] = g_t_k[L[t, k]]

  return G, L

def viterbi_ml(y):
  G, L = viterbi(y)
  Lmax = G[-1].argmax()
  T = len(y)
  R = np.zeros(T).astype(int)
  T -= 1
  R[T] = Lmax

  while T > 0:
    # whew.
    # L encodes "best previous state" for all states at time t, (T, K)
    # We select our best estimate, R[T - 1]
    # by taking best estimate at R[T], and finding the best previous candidate
    # L[T, R[T]]
    R[T - 1] = L[T, R[T]]
    T -= 1

  return R

##### Maximum-likelihood with no time-series information #####

def baseline_model(ys):
  ys = np.array(ys)
  x  = (ys > (mu1 / 2)).astype(int)
  return x

##### Testing #####

def test(model, num_obs, *args, **kwargs):
  (x, y) = simulate_forward(num_obs)
  x_ml   = model(y, *args, **kwargs)
  return (x, x_ml)

def test_baseline(num_obs):
  return test(baseline_model, num_obs)

def test_bootstrap(num_obs):
  return test(bootstrap_ml, num_obs, num_particles=1000)

def test_viterbi(num_obs):
  return test(viterbi_ml, num_obs)

def evaluate(test_fun, n=10, *args, **kwargs):
  for i in range(0, n):
    x, x_ml = test_fun(*args, **kwargs)

    accuracy = 1 - np.abs(x - x_ml).sum() / len(x)
    yield accuracy

if __name__ == "__main__":
  settings = dict(
    n = 1000,
    num_obs = 40,
  )

  run_test = lambda f: np.mean(list(evaluate(f, **settings) ))

  print('baseline : ', run_test(test_baseline))
  print('viterbi  : ', run_test(test_viterbi))
  print('bootstrap: ', run_test(test_bootstrap))
