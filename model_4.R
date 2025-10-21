# === Load Libraries ===
library(cmdstanr)
library(posterior)
library(tidyr)
options(mc.cores = parallel::detectCores())
#Simulate data
N <- 100
T <- 20
K <- 4
matrix <- matrix(0, nrow = N, ncol = T)
true_group_assignments <- numeric(N)

for (i in 1:N) {
  group <- sample(1:K, 1)
  true_group_assignments[i] <- group
  e <- rnorm(T, 0, 0.05)
  x <- numeric(T)
  if (group == 1) {
    x[1] <- 1; for (t in 2:T) x[t] <- 1 + 0.1 * x[t - 1] + e[t]
  } else if (group == 2) {
    x[1] <- 1; for (t in 2:T) x[t] <- 0 + 0.8 * x[t - 1] + e[t]
  } else if (group == 3) {
    x[1] <- 0; for (t in 2:T) x[t] <- 0.1 + 0.92 * x[t - 1] + e[t]
  } else {
    x[1] <- 0; for (t in 2:T) x[t] <- 0 + 0.1 * x[t - 1] + e[t]
  }
  matrix[i, ] <- x
}
#Quick plot of generated means to be able to compare to the output
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

# === Stan Model Code ===
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
  sigma ~ normal(0, 0.05);
  theta ~ dirichlet(rep_vector(1.0, 4));

  mu_alpha ~ normal([1, 0.9, 0.1, 0], 0.05);
  sigma_alpha ~ normal(0, 0.05) T[0,];

  mu_rho ~ normal([0.1, 0.8, 0.92, 0.1], 0.05);
  sigma_rho ~ normal(0, 0.05) T[0,];

  beta ~ normal([1, 0, 0, 0], 0.05);

  alpha_raw ~ normal(0, 1);
  rho_raw ~ normal(0, 1);

  for (n in 1:N) {
    vector[4] log_lik;

    for (k in 1:4) {
      real alpha_n = mu_alpha[k] + sigma_alpha[k] * alpha_raw[n];
      real rho_n   = mu_rho[k] + sigma_rho[k] * rho_raw[n];

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
      real rho_n   = mu_rho[k] + sigma_rho[k] * rho_raw[n];

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

# Fit Model
stan_file <- write_stan_file(stan_code)
model <- cmdstan_model(stan_file)
data_list <- list(N = N, T = T, x = matrix)

fit <- model$sample(
  data = data_list,
  chains = 4,
  iter_warmup = 150,
  iter_sampling = 150,
  parallel_chains = 4,
  adapt_delta = 0.99,
  max_treedepth = 25,
  refresh = 50
)

# === Diagnostics ===
summary_df <- fit$summary()
summary_df
# === Posterior Class Probabilities ===
samples <- fit$draws(variables = "k_assign", format = "draws_array")
k_samples <- as_draws_matrix(samples)
posterior_probs <- matrix(0, nrow = N, ncol = K)
for (i in 1:N) for (k in 1:K) posterior_probs[i, k] <- mean(k_samples[, i] == k)

# === Modal Assignment & Accuracy ===
mode_assign <- apply(k_samples, 2, function(x) {
  tab <- table(x)
  as.integer(names(tab)[which.max(tab)])
})

accuracy <- mean(mode_assign == true_group_assignments)
cat("Classification Accuracy:", round(accuracy * 100, 2), "%\n")

comparison_df <- data.frame(
  True_Group = true_group_assignments,
  Inferred_Group = mode_assign
)
comparison_df

#Posterior Predictive Checks (Group Means)
post <- as_draws_df(fit$draws(variables = c("mu_alpha", "sigma_alpha",
                                            "mu_rho", "sigma_rho",
                                            "sigma", "beta", "k_assign",
                                            "alpha_raw", "rho_raw")))
sample_row <- sample(nrow(post), 1)
s <- post[sample_row, ]

extract_vec <- function(name) sapply(1:K, function(k) as.numeric(s[[paste0(name, "[", k, "]")]]))
mu_alpha <- extract_vec("mu_alpha")
sigma_alpha <- extract_vec("sigma_alpha")
mu_rho <- extract_vec("mu_rho")
sigma_rho <- extract_vec("sigma_rho")
beta <- extract_vec("beta")
sigma <- as.numeric(s[["sigma"]])
alpha_raw <- sapply(1:N, function(i) as.numeric(s[[paste0("alpha_raw[", i, "]")]]))
rho_raw <- sapply(1:N, function(i) as.numeric(s[[paste0("rho_raw[", i, "]")]]))
k_assign <- sapply(1:N, function(i) as.numeric(s[[paste0("k_assign[", i, "]")]]))

predicted_matrix <- matrix(NA, N, T)
for (i in 1:N) {
  g <- k_assign[i]
  alpha_i <- mu_alpha[g] + sigma_alpha[g] * alpha_raw[i]
  rho_i <- mu_rho[g] + sigma_rho[g] * rho_raw[i]
  y <- numeric(T); y[1] <- rnorm(1, alpha_i, sigma)
  for (t in 2:T) y[t] <- rnorm(1, beta[g] + rho_i * y[t-1], sigma)
  predicted_matrix[i, ] <- y
}

group_colors <- c("blue", "red", "green", "orange")
mean_trajectories <- sapply(1:K, function(g) {
  rows <- predicted_matrix[k_assign == g, , drop = FALSE]
  if (nrow(rows) > 0) colMeans(rows) else rep(NA, T)
})

plot(NULL, xlim = c(1, T), ylim = range(mean_trajectories, na.rm = TRUE),
     xlab = "Time", ylab = "Value", main = "Posterior Predictive Mean Trajectories")
for (g in 1:K) {
  if (all(is.finite(mean_trajectories[, g]))) {
    lines(1:T, mean_trajectories[, g], col = group_colors[g], lwd = 2)
  }
}
legend("topright", legend = paste("Group", 1:K),
       col = group_colors, lty = 1, lwd = 2, cex = 0.8)

