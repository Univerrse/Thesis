# Baseline Growth Mixture Model (no AR and no Random effects)

library(cmdstanr)
library(posterior)
library(bayesplot)
options(mc.cores = parallel::detectCores())

# ------------------------------------------------------------------
# Data Generation: simulate data with 1 "troublemaker" high-variance group

N <- 100
T <- 20
K <- 4
matrix <- matrix(0, nrow = N, ncol = T)
true_group_assignments <- integer(N)

# Variance settings: group 3 will be the "troublemaker"
group_sigmas <- c(0.05, 0.05, 0.25, 0.05)  # high variance for group 4

for (i in 1:N) {
  group <- sample(1:K, 1)
  true_group_assignments[i] <- group
  e <- rnorm(T, 0, group_sigmas[group])
  
  x <- numeric(T)
  if (group == 1) {
    x[1] <- 1
    for (t in 2:T) x[t] <- 1 + 0.1 * x[t - 1] + e[t]
  } else if (group == 2) {
    x[1] <- 1
    for (t in 2:T) x[t] <- 1 - 0.05 * t + 0.1 * x[t - 1] + e[t]
  } else if (group == 3) {
    x[1] <- 0
    for (t in 2:T) x[t] <- 0 + 0.05 * t + 0.1 * x[t - 1] + e[t]
  } else {
    x[1] <- 0
    for (t in 2:T) x[t] <- 0 + 0.1 * x[t - 1] + e[t]
  }
  matrix[i, ] <- x
}


# Quick plot of generated means (by true group)
group_colors <- c("blue", "red", "green", "orange")
group_means <- sapply(1:4, function(g) colMeans(matrix[true_group_assignments == g, , drop = FALSE]))
plot(1:T, group_means[,1], type="l", col=group_colors[1], ylim=range(matrix),
     ylab="Value", xlab="Time", main="Mean Trajectories of 4 Groups (True)")
for (g in 2:4) lines(1:T, group_means[,g], col=group_colors[g], lwd=2)
legend("topright",
       legend=c("Group 1 ~1", "Group 2 dec 1→0", "Group 3 inc 0→1", "Group 4 ~0"),
       col=group_colors, lty=1, lwd=2, cex=0.9)

# ------------------------------------------------------------------
# Stan model code: BASIC GROWTH MIXTURE (no AR, linear growth per class)
# x_{n,t} | z_n=k ~ Normal(beta0[k] + beta1[k] * t[tt], sigma[k])
# z_n ~ Categorical(theta)
K <- 4
t_raw <- 0:(T-1)
# Center time to reduce correlation between intercept & slope:
t <- as.vector(scale(t_raw, center = TRUE, scale = FALSE))

stan_code <- '
data {
  int<lower=1> N;            // subjects
  int<lower=2> T;            // time points
  matrix[N, T] x;            // data: rows are subjects
  vector[T] t;               // time index
  int<lower=1> K;            // number of classes
}
parameters {
  simplex[K] theta;
  vector[K] beta0;                 // intercept per class
  vector[K] beta1;                 // slope per class
  vector<lower=0.001>[K] sigma;     // residual SD per class
}
model {
  // Class 1: stays near 1  -> intercept ~ 1, slope ~ 0
  beta0[1] ~ normal(1.0, 0.1);
  beta1[1] ~ normal(0.0, 0.1);

  // Class 2: decreases 1 -> 0 -> small negative slope, start near 1
  beta0[2] ~ normal(1.0, 0.05);
  beta1[2] ~ normal(-0.07, 0.05);

  // Class 3: increases 0 -> 1 -> small positive slope, start near 0
  beta0[3] ~ normal(0.0, 0.05);
  beta1[3] ~ normal(0.07, 0.05);

  // Class 4: stays near 0 -> intercept ~ 0, slope ~ 0
  beta0[4] ~ normal(0.0, 0.05);
  beta1[4] ~ normal(0.0, 0.05);

  // Residual SDs & class weights
  sigma ~ normal(0, 0.05);                 // > 0 due to lower bound
  theta ~ dirichlet(rep_vector(2.0, K)); // mildly favors using all classes Should I do it? Ask Nalan

  // Likelihood
  for (n in 1:N) {
    vector[K] lps;
    for (k in 1:K) {
      real ll = 0;
      for (tt in 1:T) {
        real mu = beta0[k] + beta1[k] * t[tt];
        ll += normal_lpdf(x[n, tt] | mu, sigma[k]);
      }
      lps[k] = log(theta[k]) + ll;
    }
    target += log_sum_exp(lps);
  }
}
generated quantities {
  // hard class draw per subject + one posterior predictive draw
  array[N] int<lower=1, upper=K> k_assign;
  matrix[N, T] x_rep;

  for (n in 1:N) {
    vector[K] lps;
    for (k in 1:K) {
      real ll = 0;
      for (tt in 1:T) {
        real mu = beta0[k] + beta1[k] * t[tt];
        ll += normal_lpdf(x[n, tt] | mu, sigma[k]);
      }
      lps[k] = log(theta[k]) + ll;
    }
    k_assign[n] = categorical_logit_rng(lps);

    for (tt in 1:T) {
      real mu = beta0[k_assign[n]] + beta1[k_assign[n]] * t[tt];
      x_rep[n, tt] = normal_rng(mu, sigma[k_assign[n]]);
    }
  }
}
'

# ------------------------------------------------------------------
# Compile & fit
stan_file <- write_stan_file(stan_code)
model <- cmdstan_model(stan_file)

data_list <- list(N = N, T = T, x = matrix, t = t, K = K)

fit <- model$sample(
  data = data_list,
  chains = 4,
  parallel_chains = parallel::detectCores(),
  iter_warmup = 150,
  iter_sampling = 150,
  adapt_delta = 0.98,
  max_treedepth = 20,
  refresh = 50
)

print(fit$summary(c("theta", "beta0", "beta1", "sigma")))

# ------------------------------------------------------------------
# Class assignments: mode over draws
draws_k <- fit$draws("k_assign", format = "draws_matrix")

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

k_cols <- grep("^k_assign\\[", colnames(draws_k))
inferred_groups <- apply(draws_k[, k_cols, drop = FALSE], 2, Mode)

# Compare to true labels (NOTE: label switching means this is "naive")
comparison <- data.frame(
  True_Group = true_group_assignments,
  Inferred_Group = inferred_groups
)
print(comparison)
cat("Naive accuracy (no label alignment):",
    mean(comparison$True_Group == comparison$Inferred_Group), "\n")

# ------------------------------------------------------------------
# Posterior predictive: class mean trajectories from one posterior draw
post_df <- as_draws_df(fit$draws(c("beta0","beta1","sigma","k_assign")))
posterior_sample <- sample(nrow(post_df), 1)

b0  <- as.numeric(post_df[posterior_sample, grep("^beta0\\[", names(post_df))])
b1  <- as.numeric(post_df[posterior_sample, grep("^beta1\\[", names(post_df))])

pred_means <- sapply(1:K, function(g) b0[g] + b1[g] * t)  # T x K

plot(1:T, pred_means[,1], type="l", col=group_colors[1],
     ylim=range(pred_means), xlab="Time", ylab="Value",
     main="Class-specific Linear Growth (Posterior Draw)")
for (g in 2:K) lines(1:T, pred_means[,g], col=group_colors[g], lwd=2)
legend("topright", legend=paste("Class", 1:K),
       col=group_colors, lty=1, lwd=2, cex=0.9)

#Entropy
# For each subject, get probability distribution over classes
n_draws <- nrow(draws_k)
post_probs <- matrix(0, N, K)

for (i in 1:N) {
  k_col_name <- paste0("k_assign[", i, "]")
  assignments <- draws_k[, k_col_name]
  for (k in 1:K) {
    post_probs[i, k] <- mean(assignments == k)
  }
}
# Entropy: H = -sum(p_k * log(p_k))
# Higher entropy = more uncertainty about class membership
entropy <- apply(post_probs, 1, function(p) {
  p <- p[p > 0]  # avoid log(0)
  -sum(p * log(p))
})
# Entropy ranges from 0 (certain) to log(K) (maximum uncertainty)
max_entropy <- log(K)
cat("Entropy range: [", round(min(entropy), 3), ",", round(max(entropy), 3), "]\n")
cat("Max possible entropy:", round(max_entropy, 3), "\n")
cat("Mean entropy:", round(mean(entropy), 3), "\n")

# Classify subjects by certainty
high_certainty <- entropy < 0.5  # threshold is somewhat arbitrary
cat("Subjects with high certainty (H < 0.5):", sum(high_certainty), "/", N, "\n")

# Compare entropy by true group
boxplot(entropy ~ true_group_assignments, 
        xlab = "True Group", 
        ylab = "Posterior Entropy",
        main = "Classification Uncertainty by True Group")
# Show subjects with highest entropy
uncertain_subjects <- order(entropy, decreasing = TRUE)[1:5]
for (i in uncertain_subjects) {
  cat("\nSubject", i, "- Entropy:", round(entropy[i], 3), "\n")
  cat("True group:", true_group_assignments[i], "\n")
  cat("Posterior probabilities:", round(post_probs[i,], 3), "\n")
}

# ------------------------------------------------------------------ 
# POSTERIOR UNCERTAINTY ANALYSIS: Entropy
# ------------------------------------------------------------------ 

# 1. Calculate posterior class probabilities for each subject
n_draws <- nrow(draws_k)
post_probs <- matrix(0, N, K)

for (i in 1:N) {
  k_col_name <- paste0("k_assign[", i, "]")
  assignments <- draws_k[, k_col_name]
  for (k in 1:K) {
    post_probs[i, k] <- mean(assignments == k)
  }
}

# 2. Calculate entropy for each subject
entropy <- apply(post_probs, 1, function(p) {
  p <- p[p > 0]  # avoid log(0)
  -sum(p * log(p))
})

# 3. Summary statistics
max_entropy <- log(K)
cat("\n=== ENTROPY ANALYSIS ===\n")
cat("Entropy range: [", round(min(entropy), 3), ",", round(max(entropy), 3), "]\n")
cat("Max possible entropy (log(K)):", round(max_entropy, 3), "\n")
cat("Mean entropy:", round(mean(entropy), 3), "\n")
cat("Median entropy:", round(median(entropy), 3), "\n")
cat("SD entropy:", round(sd(entropy), 3), "\n\n")

# Count by certainty levels
very_certain <- sum(entropy < 0.1)
certain <- sum(entropy >= 0.1 & entropy < 0.5)
uncertain <- sum(entropy >= 0.5 & entropy < 1.0)
very_uncertain <- sum(entropy >= 1.0)

cat("Classification certainty distribution:\n")
cat("  Very certain (H < 0.1):", very_certain, "/", N, "\n")
cat("  Certain (0.1 ≤ H < 0.5):", certain, "/", N, "\n")
cat("  Uncertain (0.5 ≤ H < 1.0):", uncertain, "/", N, "\n")
cat("  Very uncertain (H ≥ 1.0):", very_uncertain, "/", N, "\n\n")

# ------------------------------------------------------------------ 
# VISUALIZATIONS
# ------------------------------------------------------------------ 

# Plot 1: Histogram of entropy values
hist(entropy, breaks = 30, 
     col = "lightblue", 
     border = "black",
     main = "Distribution of Posterior Entropy Across Subjects",
     xlab = "Entropy (bits)",
     ylab = "Frequency")
abline(v = max_entropy, col = "red", lwd = 2, lty = 2)
text(max_entropy, par("usr")[4] * 0.9, 
     labels = paste0("Max entropy = ", round(max_entropy, 2)), 
     pos = 2, col = "red")

# Plot 2: Entropy by true group (boxplot)
par(mfrow = c(1, 2))
boxplot(entropy ~ true_group_assignments, 
        col = group_colors,
        xlab = "True Group", 
        ylab = "Posterior Entropy",
        main = "Entropy by True Group")
abline(h = max_entropy, col = "red", lty = 2)

# Plot 3: Entropy by inferred group
boxplot(entropy ~ inferred_groups, 
        col = group_colors,
        xlab = "Inferred Group", 
        ylab = "Posterior Entropy",
        main = "Entropy by Inferred Group")
abline(h = max_entropy, col = "red", lty = 2)

# Plot 4: Scatter plot - entropy vs subject index (colored by true group)
par(mfrow = c(1, 1))
plot(1:N, entropy, 
     col = group_colors[true_group_assignments],
     pch = 19,
     xlab = "Subject Index",
     ylab = "Posterior Entropy",
     main = "Entropy for Each Subject (colored by true group)")
abline(h = max_entropy, col = "red", lty = 2)
legend("topright", 
       legend = paste("Group", 1:K), 
       col = group_colors, 
       pch = 19, 
       cex = 0.8)

# ------------------------------------------------------------------ 
# DETAILED EXAMINATION OF UNCERTAIN CASES
# ------------------------------------------------------------------ 

# Show the 10 most uncertain subjects
n_show <- min(10, N)
uncertain_subjects <- order(entropy, decreasing = TRUE)[1:n_show]

cat("\n=== TOP", n_show, "MOST UNCERTAIN SUBJECTS ===\n")
for (i in uncertain_subjects) {
  cat("\nSubject", i, "- Entropy:", round(entropy[i], 3), "\n")
  cat("  True group:", true_group_assignments[i], "\n")
  cat("  Inferred group (mode):", inferred_groups[i], "\n")
  cat("  Posterior probabilities: [", 
      paste(sapply(1:K, function(k) 
        paste0("Class ", k, ": ", sprintf("%.3f", post_probs[i, k]))), 
        collapse = ", "), 
      "]\n")
}

# Show the 10 most certain subjects
certain_subjects <- order(entropy, decreasing = FALSE)[1:n_show]

cat("\n=== TOP", n_show, "MOST CERTAIN SUBJECTS ===\n")
for (i in certain_subjects) {
  cat("\nSubject", i, "- Entropy:", round(entropy[i], 3), "\n")
  cat("  True group:", true_group_assignments[i], "\n")
  cat("  Inferred group (mode):", inferred_groups[i], "\n")
  cat("  Posterior probabilities: [", 
      paste(sapply(1:K, function(k) 
        paste0("Class ", k, ": ", sprintf("%.3f", post_probs[i, k]))), 
        collapse = ", "), 
      "]\n")
}

# ------------------------------------------------------------------ 
# VISUAL: Plot trajectories of most uncertain subjects
# ------------------------------------------------------------------ 

par(mfrow = c(2, 3))
for (idx in 1:min(6, length(uncertain_subjects))) {
  i <- uncertain_subjects[idx]
  plot(1:T, matrix[i, ], 
       type = "b", 
       pch = 19,
       col = group_colors[true_group_assignments[i]],
       ylim = range(matrix),
       main = paste0("Subject ", i, " (H=", round(entropy[i], 2), ")"),
       xlab = "Time",
       ylab = "Value")
  
  # Add posterior probabilities as text
  prob_text <- paste0("P(k): ", paste(round(post_probs[i,], 2), collapse = ", "))
  mtext(prob_text, side = 3, line = 0, cex = 0.7)
  
  # Overlay class mean trajectories
  for (g in 1:K) {
    lines(1:T, pred_means[, g], col = group_colors[g], lty = 2, lwd = 1)
  }
}
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

cat("\n=== FULL ENTROPY TABLE (sorted by entropy, showing top 20) ===\n")
print(head(entropy_table, 20))

# Correlation between entropy and being correct
cat("\n=== RELATIONSHIP: Entropy vs Classification Accuracy ===\n")
cat("Mean entropy when CORRECT:", 
    round(mean(entropy[true_group_assignments == inferred_groups]), 3), "\n")
cat("Mean entropy when INCORRECT:", 
    round(mean(entropy[true_group_assignments != inferred_groups]), 3), "\n")

# Optional: save full table to CSV
# write.csv(entropy_table, "entropy_analysis.csv", row.names = FALSE)


