def bootstrap(ys, num_particles):
  # 1. Initialization
  T       = len(ys)
  # A grid T "tall", and num_particles "wide".
  X       = np.zeros(shape=(T, num_particles))
  # draw from x0 N times.
  X[0]    = draw_x0(num_particles)

  indexes = np.array(range(0, num_particles))

  # weight each particle based on probability of observing y_0
  # then normalise + select
  wt_tmp = normalise( py_bar_xt( ys[0], X[0] ) )
  ixs    = r.choice(indexes, p=wt_tmp, size=num_particles)
  X      =  X[:, ixs]

  for t in range(1, T):
    # 2. Importance sampling step
    X[t]   = draw_transition(X[t - 1])
    wt_tmp = normalise( py_bar_xt( ys[t], X[t] ) )

    # 3. Selection step: pick particles weighted by importance.
    ixs = r.choice(indexes, p=wt_tmp, size=num_particles)
    X = X[:, ixs]

  return X
