# Random-Effects GMM with AR(1), troublemaker case and Entropy analysis
library(cmdstanr)
library(posterior)
library(bayesplot)

options(mc.cores = parallel::detectCores())

#Simulate data
N <- 500
T <- 20
K <- 4
X <- matrix(0, nrow = N, ncol = T)
true_group_assignments <- integer(N)

# NOTE: high variance here is for group 2
group_sigmas <- c(0.05, 0.25, 0.05, 0.05)

for (i in 1:N) {
  g <- sample(1:K, 1)
  true_group_assignments[i] <- g
  e <- rnorm(T, 0, group_sigmas[g])
  
  x <- numeric(T)
  if (g == 1) {
    x[1] <- 1
    for (t in 2:T) x[t] <- 1 + 0.1 * x[t - 1] + e[t]
  } else if (g == 2) {
    x[1] <- 1
    for (t in 2:T) x[t] <- 1 - 0.05 * t + 0.1 * x[t - 1] + e[t]
  } else if (g == 3) {
    x[1] <- 0
    for (t in 2:T) x[t] <- 0 + 0.05 * t + 0.1 * x[t - 1] + e[t]
  } else {
    x[1] <- 0
    for (t in 2:T) x[t] <- 0 + 0.1 * x[t - 1] + e[t]
  }
  X[i, ] <- x
}

# Quick plot of generated means (by true group)
group_colors <- c("blue", "red", "green", "orange")
group_means <- sapply(1:K, function(g) {
  rows <- X[true_group_assignments == g, , drop = FALSE]
  if (nrow(rows) > 0) colMeans(rows) else rep(NA_real_, T)
})

plot(1:T, group_means[, 1], type = "l", col = group_colors[1], ylim = range(X),
     ylab = "Value", xlab = "Time", main = "Mean Trajectories of 4 True Groups", lwd = 2)
for (g in 2:K) if (all(is.finite(group_means[, g]))) lines(1:T, group_means[, g], col = group_colors[g], lwd = 2)
legend("topright",
       legend = c("G1: stays at 1", "G2: dec 1→0", "G3: inc 0→1", "G4: stays at 0"),
       col = group_colors, lty = 1, lwd = 2, bty = "n")

# -------------------------
# Stan model (random effects + mixture)
# IMPORTANT: avoid vector literal priors; do elementwise
# -------------------------
stan_code <- '
data {
  int<lower=1> N;
  int<lower=1> T;
  matrix[N, T] x;
}

parameters {
  real<lower=0.0001> sigma;
  simplex[4] theta;

  vector[4] mu_alpha;
  vector<lower=0.0001>[4] sigma_alpha;

  vector[4] mu_rho;
  vector<lower=0.0001>[4] sigma_rho;

  vector[4] beta;

  vector[N] alpha_raw;
  vector[N] rho_raw;
}

model {
  // hyperpriors & priors
  sigma ~ normal(0, 0.1);
  theta ~ dirichlet(rep_vector(1.0, 4));

  mu_alpha[1] ~ normal(1.0, 0.05);
  mu_alpha[2] ~ normal(0.9, 0.05);
  mu_alpha[3] ~ normal(0.1, 0.05);
  mu_alpha[4] ~ normal(0.0, 0.05);
  sigma_alpha ~ normal(0.05, 0.02) T[0,];

  mu_rho[1] ~ normal(0.1, 0.05);
  mu_rho[2] ~ normal(0.8, 0.05);
  mu_rho[3] ~ normal(0.92, 0.05);
  mu_rho[4] ~ normal(0.1, 0.05);
  sigma_rho ~ normal(0.05, 0.02) T[0,];

  beta[1] ~ normal(1.0, 0.05);
  beta[2] ~ normal(0.0, 0.05);
  beta[3] ~ normal(0.0, 0.05);
  beta[4] ~ normal(0.0, 0.05);

  alpha_raw ~ normal(0, 1);
  rho_raw   ~ normal(0, 1);

  // mixture likelihood with subject-level REs under each class
  for (n in 1:N) {
    vector[4] log_lik;
    for (k in 1:4) {
      real alpha_n = mu_alpha[k] + sigma_alpha[k] * alpha_raw[n];
      real rho_n   = mu_rho[k]   + sigma_rho[k]   * rho_raw[n];

      real ll = normal_lpdf(x[n,1] | alpha_n, sigma);
      for (t in 2:T) {
        real mu = beta[k] + rho_n * x[n,t-1];
        ll += normal_lpdf(x[n,t] | mu, sigma);
      }
      log_lik[k] = ll;
    }
    target += log_sum_exp(log(theta) + log_lik);
  }
}

generated quantities {
  array[N] int<lower=1, upper=4> k_assign;
  for (n in 1:N) {
    vector[4] log_lik;
    for (k in 1:4) {
      real alpha_n = mu_alpha[k] + sigma_alpha[k] * alpha_raw[n];
      real rho_n   = mu_rho[k]   + sigma_rho[k]   * rho_raw[n];

      real ll = normal_lpdf(x[n,1] | alpha_n, sigma);
      for (t in 2:T) {
        real mu = beta[k] + rho_n * x[n,t-1];
        ll += normal_lpdf(x[n,t] | mu, sigma);
      }
      log_lik[k] = ll;
    }
    k_assign[n] = categorical_logit_rng(log(theta) + log_lik);
  }
}
'

stan_file <- write_stan_file(stan_code)
model <- cmdstan_model(stan_file)

# -------------------------
# Sample
# -------------------------
data_list <- list(N = N, T = T, x = X)

fit <- model$sample(
  data = data_list,
  chains = 4,
  iter_warmup = 150,
  iter_sampling = 150,
  parallel_chains = min(4, parallel::detectCores()),
  adapt_delta = 0.95,
  max_treedepth = 20,
  refresh = 50
)

print(fit$summary(variables = c("mu_alpha", "sigma_alpha", "mu_rho", "sigma_rho", "sigma", "beta")))

# -------------------------
# Posterior hard assignments (mode)
# -------------------------
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

draws_k_arr <- fit$draws(variables = "k_assign", format = "draws_array")  # iters x chains x N
inferred_groups <- apply(draws_k_arr, 3, Mode)

comparison <- data.frame(
  True_Group = true_group_assignments,
  Inferred_Group = inferred_groups
)
print(head(comparison, 10))
cat("Classification Accuracy:", round(mean(comparison$True_Group == comparison$Inferred_Group) * 100, 2), "%\n")

# -------------------------
# Posterior predictive (one random posterior draw)
# Reconstruct subject-specific alpha_i, rho_i for their sampled class
# -------------------------
post_df <- as.data.frame(as_draws_df(
  fit$draws(variables = c("mu_alpha", "sigma_alpha",
                          "mu_rho", "sigma_rho",
                          "sigma", "beta", "k_assign",
                          "alpha_raw", "rho_raw"))
))
posterior_sample <- sample(nrow(post_df), 1)
s <- post_df[posterior_sample, ]

extract_vec <- function(df_row, prefix, K = 4) {
  as.numeric(df_row[1, grep(paste0("^", prefix, "\\["), names(df_row))])
}

mu_alpha  <- extract_vec(s, "mu_alpha")
sigma_alpha <- extract_vec(s, "sigma_alpha")
mu_rho    <- extract_vec(s, "mu_rho")
sigma_rho <- extract_vec(s, "sigma_rho")
beta      <- extract_vec(s, "beta")
sig       <- as.numeric(s[["sigma"]])
k_sampled <- sapply(1:N, function(i) as.numeric(s[[paste0("k_assign[", i, "]")]]))
alpha_raw <- sapply(1:N, function(i) as.numeric(s[[paste0("alpha_raw[", i, "]")]]))
rho_raw   <- sapply(1:N, function(i) as.numeric(s[[paste0("rho_raw[", i, "]")]]))

predicted_matrix <- matrix(NA_real_, N, T)
for (i in 1:N) {
  g <- k_sampled[i]
  alpha_i <- mu_alpha[g] + sigma_alpha[g] * alpha_raw[i]
  rho_i   <- mu_rho[g]   + sigma_rho[g]   * rho_raw[i]
  
  y <- numeric(T)
  y[1] <- rnorm(1, alpha_i, sig)
  for (t in 2:T) {
    mu <- beta[g] + rho_i * y[t - 1]
    y[t] <- rnorm(1, mu, sig)
  }
  predicted_matrix[i, ] <- y
}

# Group mean posterior predictive trajectories (by sampled classes)
pred_means <- sapply(1:K, function(g) {
  rows <- predicted_matrix[k_sampled == g, , drop = FALSE]
  if (nrow(rows) > 0) colMeans(rows) else rep(NA_real_, T)
})

plot(NULL, xlim = c(1, T), ylim = range(pred_means, na.rm = TRUE),
     xlab = "Time", ylab = "Value", main = "Posterior Predictive Mean Trajectories")
for (g in 1:K) if (all(is.finite(pred_means[, g]))) lines(1:T, pred_means[, g], col = group_colors[g], lwd = 2)
legend("topright", legend = paste("Group", 1:K), col = group_colors, lty = 1, lwd = 2, bty = "n")

# -------------------------
# Posterior class probabilities & Entropy
# -------------------------
draws_k_df <- as.data.frame(as_draws_df(fit$draws(variables = "k_assign")))
post_probs <- matrix(0, N, K)
for (i in 1:N) {
  k_col <- paste0("k_assign[", i, "]")
  assignments <- as.integer(draws_k_df[[k_col]])
  post_probs[i, ] <- tabulate(assignments, nbins = K) / length(assignments)
}

# Entropy (nats)
entropy <- apply(post_probs, 1, function(p) {
  p <- p[p > 0]
  -sum(p * log(p))
})
max_entropy <- log(K)

cat("Entropy (nats) range: [", round(min(entropy), 3), ", ", round(max(entropy), 3), "]\n", sep = "")
cat("Max possible (nats): ", round(max_entropy, 3), "\n", sep = "")
cat("Mean entropy (nats): ", round(mean(entropy), 3), "\n", sep = "")

# Certainty buckets
very_certain <- sum(entropy < 0.1)
certain      <- sum(entropy >= 0.1 & entropy < 0.5)
uncertain    <- sum(entropy >= 0.5 & entropy < 1.0)
very_uncertain <- sum(entropy >= 1.0)
cat("Certainty counts: very certain=", very_certain,
    ", certain=", certain, ", uncertain=", uncertain,
    ", very uncertain=", very_uncertain, "\n", sep = "")

# Visuals
hist(entropy, breaks = 20, col = "lightblue", border = "black",
     main = "Distribution of Posterior Entropy (nats)",
     xlab = "Entropy (nats)", ylab = "Frequency")
abline(v = max_entropy, col = "red", lty = 2, lwd = 2)

par(mfrow = c(1, 2))
boxplot(entropy ~ true_group_assignments, col = group_colors,
        xlab = "True Group", ylab = "Entropy (nats)", main = "Entropy by TRUE Group")
abline(h = max_entropy, col = "red", lty = 2)

boxplot(entropy ~ inferred_groups, col = group_colors,
        xlab = "Inferred Group (mode)", ylab = "Entropy (nats)", main = "Entropy by INFERRED Group")
abline(h = max_entropy, col = "red", lty = 2)

par(mfrow = c(1, 1))
plot(1:N, entropy, col = group_colors[true_group_assignments], pch = 19,
     xlab = "Subject Index", ylab = "Entropy (nats)",
     main = "Entropy per Subject (colored by TRUE group)")
abline(h = max_entropy, col = "red", lty = 2)
legend("topright", legend = paste("Group", 1:K), col = group_colors, pch = 19, bty = "n")

# Detailed lists
n_show <- min(10, N)
uncertain_subjects <- order(entropy, decreasing = TRUE)[1:n_show]
certain_subjects   <- order(entropy, decreasing = FALSE)[1:n_show]

cat("\n=== TOP ", n_show, " MOST UNCERTAIN SUBJECTS ===\n", sep = "")
for (i in uncertain_subjects) {
  cat("\nSubject ", i, " - Entropy (nats): ", round(entropy[i], 3), "\n", sep = "")
  cat("  True group: ", true_group_assignments[i], "\n", sep = "")
  cat("  Inferred group (mode): ", inferred_groups[i], "\n", sep = "")
  cat("  Posterior probabilities: [",
      paste(sapply(1:K, function(k) paste0("Class ", k, ": ", sprintf("%.3f", post_probs[i, k]))),
            collapse = ", "),
      "]\n", sep = "")
}

cat("\n=== TOP ", n_show, " MOST CERTAIN SUBJECTS ===\n", sep = "")
for (i in certain_subjects) {
  cat("\nSubject ", i, " - Entropy (nats): ", round(entropy[i], 3), "\n", sep = "")
  cat("  True group: ", true_group_assignments[i], "\n", sep = "")
  cat("  Inferred group (mode): ", inferred_groups[i], "\n", sep = "")
  cat("  Posterior probabilities: [",
      paste(sapply(1:K, function(k) paste0("Class ", k, ": ", sprintf("%.3f", post_probs[i, k]))),
            collapse = ", "),
      "]\n", sep = "")
}

#Plot trajectories of the most uncertain subjects with posterior predictive means
par(mfrow = c(2, 3))
for (idx in 1:min(6, length(uncertain_subjects))) {
  i <- uncertain_subjects[idx]
  plot(1:T, X[i, ], type = "b", pch = 19,
       col = group_colors[true_group_assignments[i]],
       ylim = range(X),
       main = paste0("Subject ", i, " (H=", round(entropy[i], 2), " nats)"),
       xlab = "Time", ylab = "Value")
  prob_text <- paste0("P(k): ", paste(round(post_probs[i, ], 2), collapse = ", "))
  mtext(prob_text, side = 3, line = 0, cex = 0.7)
  for (g in 1:K) if (all(is.finite(pred_means[, g]))) lines(1:T, pred_means[, g], col = group_colors[g], lty = 2, lwd = 1.5)
}
par(mfrow = c(1, 1))

