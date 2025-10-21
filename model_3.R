# Growth Mixture with AR components


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
    for (t in 2:T) x[t] <- 1 + 0.1 * x[t - 1] + e[t] #Stays around 1
  } else if (group == 2) {
    x <- numeric(T); x[1] <- 1
    for (t in 2:T) x[t] <- 0.8 * x[t - 1] + e[t] #Decreases from 1 to 0
  } else if (group == 3) {
    x <- numeric(T); x[1] <- 0
    for (t in 2:T) x[t] <- 0.1 + 0.92 * x[t - 1] + e[t] #Increases from 0 to 1  
  } else {
    x <- numeric(T); x[1] <- 0
    for (t in 2:T) x[t] <- 0 + 0.1 * x[t - 1] + e[t] #Stays around 0
  }
  matrix[i, ] <- x
}

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

# ------------------------------------------------------------------
# Stan model code (same model, cmdstanr format)

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

# ------------------------------------------------------------------
# Compile Stan model using cmdstanr

stan_file <- write_stan_file(stan_code)
model <- cmdstan_model(stan_file)

# ------------------------------------------------------------------
# Fit the model using cmdstanr (fully parallelized)

data_list <- list(N=N, T=T, x=matrix)

fit <- model$sample(
  data = data_list,
  chains = 4,
  parallel_chains = parallel::detectCores(),
  iter_warmup = 150,
  iter_sampling = 150,
  adapt_delta = 0.99,
  refresh = 30,
  max_treedepth = 25,
)

# Summarize results
print(fit$summary(c("sigma", "theta", "alpha", "rho")))

# Extract samples
samples <- fit$draws(variables = "k_assign", format = "draws_array")

# Compute mode for inferred groups
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

inferred_groups <- apply(samples, 3, Mode)

comparison <- data.frame(
  True_Group = true_group_assignments,
  Inferred_Group = inferred_groups
)

print(comparison)
accuracy <- mean(comparison$True_Group == comparison$Inferred_Group)
cat("Accuracy:", accuracy, "\n")

# ------------------------------------------------------------------
# Posterior Predictive Plot

post <- fit$draws(variables = c("alpha", "rho", "sigma","beta", "k_assign"))
posterior_sample <- sample(1:dim(post)[1], 1)

post_df <- as_draws_df(fit$draws(variables = c("alpha", "rho", "sigma","beta", "k_assign")))

# Get a random posterior draw row number
posterior_sample <- sample(nrow(post_df), 1)

# Extract sampled parameters:
sampled_alpha <- as.numeric(post_df[posterior_sample, grep("^alpha\\[", names(post_df))])
sampled_rho   <- as.numeric(post_df[posterior_sample, grep("^rho\\[", names(post_df))])
sampled_sigma <- as.numeric(post_df[posterior_sample, "sigma"])
sampled_beta <- as.numeric(post_df[posterior_sample, grep("^beta\\[", names(post_df))])
sampled_k_assign <- as.numeric(post_df[posterior_sample, grep("^k_assign\\[", names(post_df))])

predicted_matrix <- matrix(0, N, T)
for (i in 1:N) {
  g <- sampled_k_assign[i]
  ar_line <- numeric(T)
  ar_line[1] <- rnorm(1, sampled_alpha[g], sampled_sigma)
  for (t in 2:T) {
    mu <- sampled_beta[g] + sampled_rho[g] * ar_line[t-1]
    ar_line[t] <- rnorm(1, mu, sampled_sigma)
  }
  predicted_matrix[i, ] <- ar_line
}

mean_trajectories <- sapply(1:4, function(g) {
  rows <- predicted_matrix[sampled_k_assign == g, , drop = FALSE]
  if (nrow(rows) > 0) {
    colMeans(rows)
  } else {
    rep(NA, T)  # Fill with NA if no samples for group g
  }
})

# Exclude NA rows when computing ylim
finite_vals <- mean_trajectories[is.finite(mean_trajectories)]

plot(NULL, xlim=c(1, T), ylim=range(finite_vals, na.rm=TRUE),
     xlab="Time", ylab="Value",
     main="Mean Posterior Predictive Trajectories")

for (g in 1:4) {
  if (all(is.finite(mean_trajectories[, g]))) {
    lines(1:T, mean_trajectories[, g], col=group_colors[g], lwd=2)
  }
}

legend("topright", legend=c("Group 1: stays at 1", "Group 2: dec 1→0",
                            "Group 3: inc 0→1", "Group 4: stays at 0"),
       col=group_colors, lty=1, lwd=3, cex=0.7)

fit$cmdstan_diagnose()
