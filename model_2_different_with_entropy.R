# Random-Effects Growth Mixture (no AR)


library(cmdstanr)
library(posterior)
library(bayesplot)
options(mc.cores = parallel::detectCores())

#Data generation
N <- 100
T <- 20
K <- 4
matrix <- matrix(0, nrow = N, ncol = T)
true_group_assignments <- integer(N)

# Variance settings: group 4 will be the "troublemaker"
group_sigmas <- c(0.05, 0.10, 0.08, 0.05)  # high variance for group 4

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

y <- matrix
# Quick plot of generated means
group_colors <- c("blue", "red", "green", "orange")
group_means <- sapply(1:4, function(g) colMeans(matrix[true_group_assignments == g, ]))

plot(1:T, group_means[,1], type="l", col=group_colors[1], ylim=range(matrix),
     ylab="Value", xlab="Time", main="Mean Trajectories of 4 Groups")
for (g in 2:4) {
  lines(1:T, group_means[,g], col=group_colors[g])
}
legend("topright",
       legend=c("Group 1: stays at 1", "Group 2: dec 1→0", 
                "Group 3: inc 0→1", "Group 4: stays at 0"),
       col=group_colors, lty=1, lwd=2)

# 2) Build design (z-scored time) for stable Cholesky
t_raw <- 0:(T-1)
t_z   <- as.numeric(scale(t_raw, center = TRUE, scale = TRUE))  # mean 0, sd 1
X     <- cbind(1, t_z)  # T x 2
# -----------------------------
# 3) Stan model
# -----------------------------
stan_code <- '
data {
  int<lower=1> N;           // subjects
  int<lower=2> T;           // time points
  int<lower=1> K;           // classes
  matrix[N, T] y;           // data (rows = subjects)
  matrix[T, 2] X;           // design: [1, z-scored time]
}
parameters {
  simplex[K] theta;                  // class weights
  matrix[K, 2] beta;                 // class mean (intercept, slope)

  // log-scales for SDs (safer than <lower=0>)
  matrix[K, 2] log_u_sd;             // SD of subj REs per class (intercept, slope)
  vector[K]    log_sigma;            // residual SD per class
}
transformed parameters {
  matrix<lower=1e-8>[K, 2] u_sd;
  vector<lower=1e-8>[K]   sigma;
  for (k in 1:K) {
    u_sd[k,1] = fmin(fmax(exp(log_u_sd[k,1]), 1e-6), 2.0);
    u_sd[k,2] = fmin(fmax(exp(log_u_sd[k,2]), 1e-6), 2.0);
    sigma[k]  = fmin(fmax(exp(log_sigma[k]),   1e-6), 2.0);
  }
}
model {
  // Anchored priors
  beta[1,1] ~ normal(1.0, 0.05);  beta[1,2] ~ normal( 0.00, 0.05); // ~1 flat
  beta[2,1] ~ normal(1.0, 0.05);  beta[2,2] ~ normal(-0.07, 0.05); // down
  beta[3,1] ~ normal(0.0, 0.05);  beta[3,2] ~ normal( 0.07, 0.05); // up
  beta[4,1] ~ normal(0.0, 0.05);  beta[4,2] ~ normal( 0.00, 0.05); // ~0 flat

  to_vector(log_u_sd) ~ normal(log(0.12), 0.5); // median ~0.12
  log_sigma           ~ normal(log(0.10), 0.5); // median ~0.10
  theta ~ dirichlet(rep_vector(2.0, K));

  // Mixture likelihood (marginal over REs)
  for (n in 1:N) {
    vector[K] lps;
    vector[T] y_n = to_vector(y[n]);
    for (k in 1:K) {
      // mean curve
      vector[T] mu_k = X * to_vector(beta[k]);

      // covariance: X diag(u_sd^2) X^T + sigma^2 I
      vector[2] u = to_vector(u_sd[k]);
      matrix[T,T] Sigma = X * diag_matrix(u .* u) * transpose(X);
      for (t in 1:T) Sigma[t,t] += square(sigma[k]);

      // symmetrize + jitter for numerical PD
      Sigma = 0.5 * (Sigma + transpose(Sigma));
      for (t in 1:T) Sigma[t,t] += 1e-9;

      // Cholesky likelihood (stable & fast)
      matrix[T,T] L = cholesky_decompose(Sigma);
      lps[k] = log(theta[k]) + multi_normal_cholesky_lpdf(y_n | mu_k, L);
    }
    target += log_sum_exp(lps);
  }
}
generated quantities {
  matrix[N,K] resp;
  array[N] int<lower=1,upper=K> k_assign;

  for (n in 1:N) {
    vector[K] lps;
    vector[T] y_n = to_vector(y[n]);
    for (k in 1:K) {
      vector[T] mu_k = X * to_vector(beta[k]);

      vector[2] u = to_vector(u_sd[k]);
      matrix[T,T] Sigma = X * diag_matrix(u .* u) * transpose(X);
      for (t in 1:T) Sigma[t,t] += square(sigma[k]);
      Sigma = 0.5 * (Sigma + transpose(Sigma));
      for (t in 1:T) Sigma[t,t] += 1e-9;

      matrix[T,T] L = cholesky_decompose(Sigma);
      lps[k] = log(theta[k]) + multi_normal_cholesky_lpdf(y_n | mu_k, L);
    }
    resp[n]     = to_row_vector(softmax(lps));
    k_assign[n] = categorical_logit_rng(lps);
  }
}
'

stan_file <- write_stan_file(stan_code)
mod <- cmdstan_model(stan_file)

data_list <- list(N = N, T = T, K = K, y = y, X = X)


fit <- mod$sample(
  data = data_list,
  chains = 4,             
  parallel_chains = parallel::detectCores(),
  iter_warmup = 150,
  iter_sampling = 150,
  adapt_delta = 0.98,
  max_treedepth = 25,
  init = make_inits,
  refresh = 50
)

print(fit$summary(c("theta","beta","log_u_sd","log_sigma")))

# -----------------------------
# 4) Compare inferred vs true classes
#    (use responsibilities' posterior means -> MAP per subject)
# -----------------------------
draws_resp <- fit$draws("resp", format = "draws_matrix")
resp_cols  <- grep("^resp\\[", colnames(draws_resp))
resp_mean  <- apply(draws_resp[, resp_cols, drop=FALSE], 2, mean)
resp_mat   <- matrix(resp_mean, nrow = N, ncol = K, byrow = TRUE)
map_class  <- max.col(resp_mat)  # MAP class per subject

comparison <- data.frame(True = true_group_assignments, MAP = map_class)
comparison
cat("Naive accuracy (no relabeling):",
    mean(comparison$True == comparison$MAP), "\n")

# -----------------------------
# 5) Plot class mean trajectories from one posterior draw
# -----------------------------
post_df <- as_draws_df(fit$draws("beta"))
row_id  <- sample(nrow(post_df), 1)
b_mat   <- matrix(NA_real_, nrow=K, ncol=2)
for (k in 1:K) {
  b_mat[k,1] <- as.numeric(post_df[row_id, paste0("beta[",k,",1]")]) # intercept
  b_mat[k,2] <- as.numeric(post_df[row_id, paste0("beta[",k,",2]")]) # slope
}
mean_curves <- sapply(1:K, function(k) X %*% b_mat[k,])

plot(1:T, mean_curves[,1], type="l", col=group_colors[1],
     ylim=range(mean_curves), xlab="Time", ylab="Value",
     main="RE-GMM (no AR): Class Mean Trajectories (Posterior Draw)")
for (k in 2:K) lines(1:T, mean_curves[,k], col=group_colors[k], lwd=2)
legend("topright", legend=paste("Class", 1:K),
       col=group_colors, lty=1, lwd=2, cex=0.9)


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
entropy_table

# Correlation between entropy and being correct
cat("\n=== RELATIONSHIP: Entropy vs Classification Accuracy ===\n")
cat("Mean entropy when CORRECT:", 
    round(mean(entropy[true_group_assignments == inferred_groups]), 3), "\n")
cat("Mean entropy when INCORRECT:", 
    round(mean(entropy[true_group_assignments != inferred_groups]), 3), "\n")


