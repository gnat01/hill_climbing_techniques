acceptance_probability <- function(delta_energy, temperature) {
  if (temperature <= 0) {
    stop("temperature must be positive")
  }
  if (delta_energy <= 0) {
    return(1.0)
  }
  exp(-delta_energy / temperature)
}

random_walk_proposal <- function(step_size) {
  if (step_size <= 0) {
    stop("step_size must be positive")
  }
  function(state) {
    state + stats::runif(1, min = -step_size, max = step_size)
  }
}

double_well_energy <- function(x) {
  (x * x - 1.0) ^ 2
}

metropolis_sampler <- function(initial_state, energy_fn, proposal_fn, temperature, steps, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  if (temperature <= 0) {
    stop("temperature must be positive")
  }
  if (steps < 0) {
    stop("steps must be non-negative")
  }

  state <- initial_state
  energy <- energy_fn(state)
  accepted_moves <- 0L

  trace <- data.frame(
    step_index = integer(steps),
    state = numeric(steps),
    energy = numeric(steps),
    accepted = logical(steps),
    proposed_state = numeric(steps),
    proposed_energy = numeric(steps),
    delta_energy = numeric(steps),
    acceptance_probability = numeric(steps)
  )

  for (step_index in seq_len(steps)) {
    proposed_state <- proposal_fn(state)
    proposed_energy <- energy_fn(proposed_state)
    delta_energy <- proposed_energy - energy
    accept_prob <- acceptance_probability(delta_energy, temperature)
    accepted <- stats::runif(1) < accept_prob

    if (accepted) {
      state <- proposed_state
      energy <- proposed_energy
      accepted_moves <- accepted_moves + 1L
    }

    trace[step_index, ] <- list(
      step_index - 1L,
      state,
      energy,
      accepted,
      proposed_state,
      proposed_energy,
      delta_energy,
      accept_prob
    )
  }

  list(
    initial_state = initial_state,
    final_state = state,
    final_energy = energy,
    temperature = temperature,
    accepted_moves = accepted_moves,
    total_steps = steps,
    acceptance_rate = if (steps == 0) 0 else accepted_moves / steps,
    trace = trace
  )
}
