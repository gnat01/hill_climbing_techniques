to_spin <- function(image01) {
  ifelse(image01 > 0, 1L, -1L)
}

from_spin <- function(spin_image) {
  ifelse(spin_image > 0, 1L, 0L)
}

flip_noise <- function(image, flip_probability, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  mask <- matrix(stats::runif(length(image)) < flip_probability, nrow = nrow(image))
  noisy <- image
  noisy[mask] <- -noisy[mask]
  noisy
}

local_field <- function(state, observation, row, col, eta, coupling) {
  height <- nrow(state)
  width <- ncol(state)
  neighbor_sum <- 0
  if (row > 1) neighbor_sum <- neighbor_sum + state[row - 1, col]
  if (row < height) neighbor_sum <- neighbor_sum + state[row + 1, col]
  if (col > 1) neighbor_sum <- neighbor_sum + state[row, col - 1]
  if (col < width) neighbor_sum <- neighbor_sum + state[row, col + 1]
  eta * observation[row, col] + coupling * neighbor_sum
}

conditional_probability_positive <- function(field, temperature) {
  1 / (1 + exp(-2 * field / temperature))
}

geometric_schedule <- function(initial_temperature, alpha) {
  function(step_index) {
    max(initial_temperature * alpha ^ step_index, 1e-12)
  }
}

geman_geman_restore <- function(observation, eta, coupling, schedule_fn, sweeps, truth = NULL, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  state <- observation
  height <- nrow(state)
  width <- ncol(state)
  trace <- data.frame(
    step_index = integer(sweeps),
    temperature = numeric(sweeps),
    pixel_accuracy = numeric(sweeps)
  )

  for (sweep in seq_len(sweeps)) {
    temperature <- schedule_fn(sweep - 1L)
    order <- sample.int(height * width)
    for (flat_idx in order) {
      row <- ((flat_idx - 1L) %/% width) + 1L
      col <- ((flat_idx - 1L) %% width) + 1L
      field <- local_field(state, observation, row, col, eta, coupling)
      p_positive <- conditional_probability_positive(field, temperature)
      state[row, col] <- ifelse(stats::runif(1) < p_positive, 1L, -1L)
    }
    accuracy <- if (is.null(truth)) NA_real_ else mean(state == truth)
    trace[sweep, ] <- list(sweep - 1L, temperature, accuracy)
  }

  list(final_state = state, trace = trace)
}
