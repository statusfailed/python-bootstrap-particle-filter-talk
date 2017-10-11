def simulate_forward(num_obs):
  X = np.zeros(num_obs) # list of states
  Y = np.zeros(num_obs) # list of observations

  # Draw initial state
  X[0] = draw_x0(1)

  # sample initial observation
  Y[0] = r.normal(loc=X[0] * mu1, scale=1)

  for i in range(1, num_obs):
    x = np.array([X[i - 1]])  # previous state
    X[i] = draw_transition(x) # random transition
    Y[i] = r.normal(loc=X[i] * mu1, scale=1) # generate observation

  return (X, Y) # set of states + observations.
