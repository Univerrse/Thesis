
# Baseline Growth Mixture Model (no AR and no Random effects)

library(cmdstanr)
library(posterior)
library(bayesplot)
options(mc.cores = parallel::detectCores())

# ------------------------------------------------------------------
# Data Generation: simulate data from 4 distinct processes
N <- 100
T <- 20

matrix <- matrix(0, nrow = N, ncol = T)
true_group_assignments <- numeric(N)

for (i in 1:N) {
  group <- sample(1:4, 1)
  true_group_assignments[i] <- group
  e <- rnorm(T, 0, 0.1)
  
  if (group == 1) {
    x <- numeric(T); x[1] <- 1
    for (t in 2:T) x[t] <- 1 + 0.1 * x[t - 1] + e[t] # stays around 1
  } else if (group == 2) {
    x <- numeric(T); x[1] <- 1
    for (t in 2:T) x[t] <- 0.8 * x[t - 1] + e[t]     # decreases 1 → 0
  } else if (group == 3) {
    x <- numeric(T); x[1] <- 0
    for (t in 2:T) x[t] <- 0.1 + 0.92 * x[t - 1] + e[t] # increases 0 → 1
  } else {
    x <- numeric(T); x[1] <- 0
    for (t in 2:T) x[t] <- 0 + 0.1 * x[t - 1] + e[t] # stays around 0
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
  vector[T] t;               // time index (centered recommended)
  int<lower=1> K;            // number of classes
}
parameters {
  simplex[K] theta;
  vector[K] beta0;                 // intercept per class
  vector[K] beta1;                 // slope per class
  vector<lower=1e-6>[K] sigma;     // residual SD per class
}
model {
  // Class 1: stays near 1  -> intercept ~ 1, slope ~ 0
  beta0[1] ~ normal(1.0, 0.05);
  beta1[1] ~ normal(0.0, 0.05);

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
  iter_warmup = 200,
  iter_sampling = 200,
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

