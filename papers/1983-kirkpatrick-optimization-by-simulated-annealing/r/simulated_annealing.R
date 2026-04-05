acceptance_probability <- function(delta_energy, temperature) {
  if (temperature <= 0) {
    stop("temperature must be positive")
  }
  if (delta_energy <= 0) {
    return(1.0)
  }
  exp(-delta_energy / temperature)
}

geometric_schedule <- function(initial_temperature, cooling_rate) {
  if (initial_temperature <= 0) {
    stop("initial_temperature must be positive")
  }
  if (cooling_rate <= 0 || cooling_rate >= 1) {
    stop("cooling_rate must lie in (0, 1)")
  }
  function(step_index) {
    initial_temperature * cooling_rate ^ step_index
  }
}

rugged_landscape_energy <- function(x) {
  0.12 * x ^ 4 - 0.9 * x ^ 2 + 0.35 * sin(6.0 * x) + 0.12 * cos(14.0 * x)
}

random_walk_proposal <- function(step_size) {
  if (step_size <= 0) {
    stop("step_size must be positive")
  }
  function(state) {
    proposed <- state + stats::runif(1, min = -step_size, max = step_size)
    min(3.0, max(-3.0, proposed))
  }
}

simulated_annealing <- function(initial_state, energy_fn, proposal_fn, schedule_fn, steps, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  if (steps < 0) {
    stop("steps must be non-negative")
  }

  state <- initial_state
  energy <- energy_fn(state)
  best_state <- state
  best_energy <- energy
  accepted_moves <- 0L
  uphill_moves_accepted <- 0L

  trace <- data.frame(
    step_index = integer(steps),
    temperature = numeric(steps),
    state = numeric(steps),
    energy = numeric(steps),
    best_state = numeric(steps),
    best_energy = numeric(steps),
    proposed_state = numeric(steps),
    proposed_energy = numeric(steps),
    delta_energy = numeric(steps),
    accepted = logical(steps),
    acceptance_probability = numeric(steps),
    uphill_move_accepted = logical(steps)
  )

  for (step_index in seq_len(steps)) {
    temperature <- schedule_fn(step_index - 1L)
    if (temperature <= 0) {
      stop("schedule produced a non-positive temperature")
    }

    proposed_state <- proposal_fn(state)
    proposed_energy <- energy_fn(proposed_state)
    delta_energy <- proposed_energy - energy
    accept_prob <- acceptance_probability(delta_energy, temperature)
    accepted <- stats::runif(1) < accept_prob

    if (accepted) {
      state <- proposed_state
      energy <- proposed_energy
      accepted_moves <- accepted_moves + 1L
      if (delta_energy > 0) {
        uphill_moves_accepted <- uphill_moves_accepted + 1L
      }
    }

    if (energy < best_energy) {
      best_energy <- energy
      best_state <- state
    }

    trace[step_index, ] <- list(
      step_index - 1L,
      temperature,
      state,
      energy,
      best_state,
      best_energy,
      proposed_state,
      proposed_energy,
      delta_energy,
      accepted,
      accept_prob,
      accepted && delta_energy > 0
    )
  }

  list(
    initial_state = initial_state,
    final_state = state,
    final_energy = energy,
    best_state = best_state,
    best_energy = best_energy,
    accepted_moves = accepted_moves,
    uphill_moves_accepted = uphill_moves_accepted,
    total_steps = steps,
    acceptance_rate = if (steps == 0) 0 else accepted_moves / steps,
    trace = trace
  )
}
