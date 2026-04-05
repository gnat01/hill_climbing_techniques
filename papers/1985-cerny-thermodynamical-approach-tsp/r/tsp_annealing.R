euclidean_distance <- function(a, b) {
  sqrt((a[1] - b[1]) ^ 2 + (a[2] - b[2]) ^ 2)
}

build_distance_matrix <- function(points) {
  n <- nrow(points)
  matrix(
    vapply(seq_len(n * n), function(idx) {
      i <- ((idx - 1) %/% n) + 1
      j <- ((idx - 1) %% n) + 1
      euclidean_distance(points[i, ], points[j, ])
    }, numeric(1)),
    nrow = n,
    byrow = TRUE
  )
}

route_length <- function(tour, distance_matrix) {
  n <- length(tour)
  total <- 0
  for (i in seq_len(n)) {
    j <- if (i == n) 1 else i + 1
    total <- total + distance_matrix[tour[i] + 1, tour[j] + 1]
  }
  total
}

random_two_opt_proposal <- function(tour) {
  idx <- sort(sample.int(length(tour), 2))
  i <- idx[1]
  j <- idx[2]
  c(tour[seq_len(i - 1)], rev(tour[i:j]), tour[(j + 1):length(tour)])
}

geometric_schedule <- function(initial_temperature, alpha) {
  function(step_index) {
    max(initial_temperature * alpha ^ step_index, 1e-12)
  }
}

tsp_simulated_annealing <- function(initial_tour, distance_matrix, schedule_fn, steps, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  tour <- initial_tour
  current_length <- route_length(tour, distance_matrix)
  best_tour <- tour
  best_length <- current_length

  trace <- data.frame(
    step_index = integer(steps),
    temperature = numeric(steps),
    route_length = numeric(steps),
    best_route_length = numeric(steps),
    accepted = logical(steps)
  )

  accepted_moves <- 0L
  for (step_index in seq_len(steps)) {
    temperature <- schedule_fn(step_index - 1L)
    proposal <- random_two_opt_proposal(tour)
    proposal_length <- route_length(proposal, distance_matrix)
    delta <- proposal_length - current_length
    accepted <- delta <= 0 || stats::runif(1) < exp(-delta / temperature)
    if (accepted) {
      tour <- proposal
      current_length <- proposal_length
      accepted_moves <- accepted_moves + 1L
    }
    if (current_length < best_length) {
      best_length <- current_length
      best_tour <- tour
    }
    trace[step_index, ] <- list(step_index - 1L, temperature, current_length, best_length, accepted)
  }

  list(
    final_tour = tour,
    best_tour = best_tour,
    final_route_length = current_length,
    best_route_length = best_length,
    acceptance_rate = accepted_moves / steps,
    trace = trace
  )
}
