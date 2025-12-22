#Growth Mixture with AR components including Entropy analysis and a troublemaker variance

# ---- Packages & setup ----
library(cmdstanr)
library(posterior)
library(bayesplot)

options(mc.cores = parallel::detectCores())

# ---- Simulate data ----
N <- 500
T <- 20
K <- 4
X <- matrix(0, nrow = N, ncol = T)
true_group_assignments <- integer(N)

group_sigmas <- c(0.05, 0.05, 0.05, 0.05)  # high variance for group 3

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
for (g in 2:K) lines(1:T, group_means[, g], col = group_colors[g], lwd = 2)
legend("topright",
       legend = c("Group 1: stays at 1", "Group 2: dec 1→0",
                  "Group 3: inc 0→1", "Group 4: stays at 0"),
       col = group_colors, lty = 1, lwd = 2, bty = "n")

# ---- Stan model ----
stan_code <- '
data {
  int<lower=1> N;
  int<lower=1> T;
  matrix[N, T] x;
}

parameters {
  real<lower=0.001> sigma;
  simplex[4] theta;
  array[4] real alpha;
  array[4] real rho;
  array[4] real beta;
}

model {
  // priors (loosely informed)
  alpha[1] ~ normal(1, 0.05);
  alpha[2] ~ normal(0.9, 0.05);
  alpha[3] ~ normal(0.1, 0.05);
  alpha[4] ~ normal(0, 0.05);

  rho[1] ~ normal(0.1, 0.05);
  rho[2] ~ normal(0.8, 0.05);
  rho[3] ~ normal(0.92, 0.05);
  rho[4] ~ normal(0.1, 0.05);

  beta[1] ~ normal(1, 0.05);
  beta[2] ~ normal(0, 0.05);
  beta[3] ~ normal(0, 0.05);
  beta[4] ~ normal(0, 0.05);

  sigma ~ normal(0, 0.1);
  theta ~ dirichlet(rep_vector(1.0, 4));

  // likelihood (mixture over classes)
  for (n in 1:N) {
    vector[4] log_lik;
    for (k in 1:4) {
      real ll = normal_lpdf(x[n,1] | alpha[k], sigma);
      for (t in 2:T) {
        real mu = beta[k] + rho[k] * x[n,t-1];
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
      real ll = normal_lpdf(x[n,1] | alpha[k], sigma);
      for (t in 2:T) {
        real mu = beta[k] + rho[k] * x[n,t-1];
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

# ---- Fit ----
data_list <- list(N = N, T = T, x = X)

fit <- model$sample(
  data = data_list,
  chains = 4,
  parallel_chains = min(4, parallel::detectCores()),
  iter_warmup = 150,
  iter_sampling = 150,
  adapt_delta = 0.95,
  refresh = 50,
  max_treedepth = 25
)

print(fit$summary(c("sigma", "theta", "alpha", "rho")))

# ---- Posterior class assignments (mode per subject) ----
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Extract k_assign as draws_array for a simple mode calculation
samples_k <- fit$draws(variables = "k_assign", format = "draws_array")
# samples_k dims: iterations x chains x N
# Collapse draws across iterations and chains for each subject
inferred_groups <- apply(samples_k, 3, Mode)

comparison <- data.frame(
  True_Group = true_group_assignments,
  Inferred_Group = inferred_groups
)
print(head(comparison, 10))
accuracy <- mean(comparison$True_Group == comparison$Inferred_Group)
cat("Accuracy:", accuracy, "\n")

# ---- Posterior Predictive (from one random posterior draw) ----
post_df <- as.data.frame(as_draws_df(
  fit$draws(variables = c("alpha", "rho", "sigma", "beta", "k_assign"))
))

posterior_sample <- sample(nrow(post_df), 1)

# Extract sampled parameters cleanly
grab_vec <- function(df, prefix) {
  as.numeric(df[posterior_sample, grep(paste0("^", prefix, "\\["), names(df))])
}
sampled_alpha <- grab_vec(post_df, "alpha")
sampled_rho   <- grab_vec(post_df, "rho")
sampled_beta  <- grab_vec(post_df, "beta")
sampled_sigma <- as.numeric(post_df[posterior_sample, "sigma"])
sampled_k_assign <- as.numeric(post_df[posterior_sample, grep("^k_assign\\[", names(post_df))])

# Simulate predicted trajectories for each subject using their sampled class
predicted_matrix <- base::matrix(0, N, T)
for (i in 1:N) {
  g <- sampled_k_assign[i]
  ar_line <- numeric(T)
  ar_line[1] <- rnorm(1, sampled_alpha[g], sampled_sigma)
  for (t in 2:T) {
    mu <- sampled_beta[g] + sampled_rho[g] * ar_line[t - 1]
    ar_line[t] <- rnorm(1, mu, sampled_sigma)
  }
  predicted_matrix[i, ] <- ar_line
}

# Posterior predictive mean trajectories by sampled class
pred_means <- sapply(1:K, function(g) {
  rows <- predicted_matrix[sampled_k_assign == g, , drop = FALSE]
  if (nrow(rows) > 0) colMeans(rows) else rep(NA_real_, T)
})

# Plot mean posterior predictive trajectories
finite_vals <- pred_means[is.finite(pred_means)]
plot(NULL, xlim = c(1, T), ylim = range(finite_vals, na.rm = TRUE),
     xlab = "Time", ylab = "Value",
     main = "Mean Posterior Predictive Trajectories")
for (g in 1:K) {
  if (all(is.finite(pred_means[, g]))) {
    lines(1:T, pred_means[, g], col = group_colors[g], lwd = 2)
  }
}
legend("topright",
       legend = c("Group 1: stays at 1", "Group 2: dec 1→0",
                  "Group 3: inc 0→1", "Group 4: stays at 0"),
       col = group_colors, lty = 1, lwd = 3, cex = 0.8, bty = "n")

# ---- Posterior uncertainty (entropy) ----
# 1) Posterior class probabilities per subject
draws_k_df <- as.data.frame(as_draws_df(fit$draws(variables = "k_assign")))
post_probs <- base::matrix(0, N, K)

for (i in 1:N) {
  k_col <- paste0("k_assign[", i, "]")
  assignments <- as.integer(draws_k_df[[k_col]])  # class label across draws
  post_probs[i, ] <- tabulate(assignments, nbins = K) / length(assignments)
}

# 2) Entropy (nats)
entropy_nats <- apply(post_probs, 1, function(p) {
  p <- p[p > 0]
  -sum(p * log(p))
})
max_entropy_nats <- log(K)

# Also in bits (optional; helpful for interpretation)
entropy_bits <- apply(post_probs, 1, function(p) {
  p <- p[p > 0]
  -sum(p * log2(p))
})
max_entropy_bits <- log2(K)

cat("Entropy (nats) range: [", round(min(entropy_nats), 3), ",", round(max(entropy_nats), 3), "]\n")
cat("Max possible (nats):", round(max_entropy_nats, 3), "\n")
cat("Mean (nats):", round(mean(entropy_nats), 3), "\n")

# Classify subjects by certainty using nats
high_certainty <- entropy_nats < 0.5
cat("Subjects with high certainty (H < 0.5 nats):", sum(high_certainty), "/", N, "\n")

# ---- Visualizations ----
# Histogram of entropy (nats)
hist(entropy_nats, breaks = 30,
     col = "lightblue", border = "black",
     main = "Distribution of Posterior Entropy Across Subjects (nats)",
     xlab = "Entropy (nats)",
     ylab = "Frequency")
abline(v = max_entropy_nats, col = "red", lwd = 2, lty = 2)
text(max_entropy_nats, par("usr")[4] * 0.9,
     labels = paste0("Max = ", round(max_entropy_nats, 2)),
     pos = 2, col = "red")

# Entropy by true group
par(mfrow = c(1, 2))
boxplot(entropy_nats ~ true_group_assignments,
        col = group_colors,
        xlab = "True Group",
        ylab = "Posterior Entropy (nats)",
        main = "Entropy by TRUE Group")
abline(h = max_entropy_nats, col = "red", lty = 2)

# Entropy by inferred group
boxplot(entropy_nats ~ inferred_groups,
        col = group_colors,
        xlab = "Inferred Group (mode)",
        ylab = "Posterior Entropy (nats)",
        main = "Entropy by INFERRED Group (mode)")
abline(h = max_entropy_nats, col = "red", lty = 2)

# Scatter: entropy vs subject index, colored by true group
par(mfrow = c(1, 1))
plot(1:N, entropy_nats,
     col = group_colors[true_group_assignments],
     pch = 19,
     xlab = "Subject Index",
     ylab = "Posterior Entropy (nats)",
     main = "Entropy for Each Subject (colored by TRUE group)")
abline(h = max_entropy_nats, col = "red", lty = 2)
legend("topright",
       legend = paste("Group", 1:K),
       col = group_colors,
       pch = 19, cex = 0.8, bty = "n")

# ---- Detailed uncertain/certain cases ----
n_show <- min(10, N)
uncertain_subjects <- order(entropy_nats, decreasing = TRUE)[1:n_show]
certain_subjects   <- order(entropy_nats, decreasing = FALSE)[1:n_show]

cat("\n=== TOP", n_show, "MOST UNCERTAIN SUBJECTS ===\n")
for (i in uncertain_subjects) {
  cat("\nSubject", i, "- Entropy (nats):", round(entropy_nats[i], 3), "\n")
  cat("  True group:", true_group_assignments[i], "\n")
  cat("  Inferred group (mode):", inferred_groups[i], "\n")
  cat("  Posterior probabilities: [",
      paste(sapply(1:K, function(k)
        paste0("Class ", k, ": ", sprintf("%.3f", post_probs[i, k]))),
        collapse = ", "),
      "]\n")
}

cat("\n=== TOP", n_show, "MOST CERTAIN SUBJECTS ===\n")
for (i in certain_subjects) {
  cat("\nSubject", i, "- Entropy (nats):", round(entropy_nats[i], 3), "\n")
  cat("  True group:", true_group_assignments[i], "\n")
  cat("  Inferred group (mode):", inferred_groups[i], "\n")
  cat("  Posterior probabilities: [",
      paste(sapply(1:K, function(k)
        paste0("Class ", k, ": ", sprintf("%.3f", post_probs[i, k]))),
        collapse = ", "),
      "]\n")
}

# ---- VISUAL: Plot trajectories of most uncertain subjects + overlay posterior predictive means 
par(mfrow = c(2, 3))
for (idx in 1:min(6, length(uncertain_subjects))) {
  i <- uncertain_subjects[idx]
  plot(1:T, X[i, ],
       type = "b", pch = 19,
       col = group_colors[true_group_assignments[i]],
       ylim = range(X),
       main = paste0("Subject ", i, " (H=", round(entropy_nats[i], 2), " nats)"),
       xlab = "Time", ylab = "Value")
  
  # Add posterior probabilities as text
  prob_text <- paste0("P(k): ", paste(round(post_probs[i, ], 2), collapse = ", "))
  mtext(prob_text, side = 3, line = 0, cex = 0.7)
  
  # Overlay posterior predictive class mean trajectories (from pred_means)
  for (g in 1:K) {
    if (all(is.finite(pred_means[, g]))) {
      lines(1:T, pred_means[, g], col = group_colors[g], lty = 2, lwd = 1.5)
    }
  }
}
X
# ============================================
# End of script
# ============================================

par(mfrow = c(1, 1))

# ------------------------------------------------------------------ 
# ENTROPY vs CLASSIFICATION ACCURACY
# ------------------------------------------------------------------ 

# Create a detailed comparison table
entropy_table <- data.frame(
  Subject = 1:N,
  True_Group = true_group_assignments,
  Inferred_Group = inferred_groups,
  Entropy = round(entropy, 3),
  Correct = true_group_assignments == inferred_groups,
  Max_Posterior_Prob = apply(post_probs, 1, max)
)

# Sort by entropy (descending)
entropy_table <- entropy_table[order(entropy_table$Entropy, decreasing = TRUE), ]

cat("\n=== FULL ENTROPY TABLE ===\n")
print(entropy_table)

# Correlation between entropy and being correct
cat("\n=== RELATIONSHIP: Entropy vs Classification Accuracy ===\n")
cat("Mean entropy when CORRECT:", 
    round(mean(entropy[true_group_assignments == inferred_groups]), 3), "\n")
cat("Mean entropy when INCORRECT:", 
    round(mean(entropy[true_group_assignments != inferred_groups]), 3), "\n")


