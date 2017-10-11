# Draw the initial state by choosing uniformly at random from {0, 1}
def draw_x0(size):
  return r.choice(xs, size=size)


# Draw a random transition from state x_t to x_{t+1}
def draw_transition(xt, p=p):
  # if "changes" is 1, then xt will switch state.
  changes = r.choice([0, 1], p=[p, 1 - p], size=len(xt))

  # XOR "xt" with "changes" to get x_{t+1}.
  xt_next = xt.astype(int) ^ changes
  return xt_next


# Probability of observation gven state
def py_bar_xt(y, x):
  return norm.pdf(y, loc=mu1*x, scale=1)
