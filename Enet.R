#############################################
# (A) DATA GENERATION 
#############################################

library(MASS)  # for mvrnorm

# 1) Example 1 data sets (Sim1)
p1    <- 8
Beta1 <- c(3, 1.5, 0, 0, 2, 0, 0, 0)
sigma1 <- 3
Sigma1 <- matrix(0, nrow=p1, ncol=p1)
for (i in 1:p1) {
  for (j in 1:p1) {
    Sigma1[i, j] <- 0.5^abs(i - j)
  }
}
Sim1_data <- function(n) {
  X <- mvrnorm(n, mu = rep(0, p1), Sigma = Sigma1)
  eps <- rnorm(n, mean = 0, sd = sigma1)
  y   <- X %*% Beta1 + eps
  data.frame(y = y, X)
}
Sim1 <- lapply(1:50, function(rep) {
  list(
    n20_1 = Sim1_data(20),  # training
    n20_2 = Sim1_data(20),  # validation
    n200  = Sim1_data(200)  # test
  )
})

# 2) Example 2 data sets (Sim2)
Beta2 <- rep(0.85, p1)
Sim2_data <- function(n) {
  X <- mvrnorm(n, mu = rep(0, p1), Sigma = Sigma1)
  eps <- rnorm(n, mean = 0, sd = sigma1)
  y   <- X %*% Beta2 + eps
  data.frame(y = y, X)
}
Sim2 <- lapply(1:50, function(rep) {
  list(
    n20_1 = Sim2_data(20),
    n20_2 = Sim2_data(20),
    n200  = Sim2_data(200)
  )
})

# 3) Example 3 data sets (Sim3)
p3    <- 40
Beta3 <- c(rep(0, 10), rep(2, 10), rep(0, 10), rep(2, 10))
sigma3 <- 15
Sigma3 <- matrix(0.5, nrow = p3, ncol = p3)
diag(Sigma3) <- 1
Sim3_data <- function(n) {
  X <- mvrnorm(n, mu = rep(0, p3), Sigma = Sigma3)
  eps <- rnorm(n, mean = 0, sd = sigma3)
  y   <- X %*% Beta3 + eps
  data.frame(y = y, X)
}
Sim3 <- lapply(1:50, function(rep) {
  list(
    n100_1 = Sim3_data(100),
    n100_2 = Sim3_data(100),
    n400   = Sim3_data(400)
  )
})

# 4) Example 4 data sets (Sim4)
p4     <- 40
Beta4  <- c(rep(3, 15), rep(0, 25))
sigma4 <- 15
Sim4_data <- function(n) {
  X <- matrix(0, nrow = n, ncol = p4)
  for (i in 1:n) {
    Z1 <- rnorm(1, 0, 1)
    Z2 <- rnorm(1, 0, 1)
    Z3 <- rnorm(1, 0, 1)
    e_x <- rnorm(15, 0, 0.01)
    # fill first 15 columns
    for (j in 1:5) {
      X[i, j] <- Z1 + e_x[j]
    }
    for (j in 6:10) {
      X[i, j] <- Z2 + e_x[j]
    }
    for (j in 11:15) {
      X[i, j] <- Z3 + e_x[j]
    }
    # columns 16..40 ~ N(0,1)
    X[i, 16:p4] <- rnorm(p4 - 15, 0, 1)
  }
  e <- rnorm(n, mean = 0, sd = sigma4)
  y <- X %*% Beta4 + e
  data.frame(y = y, X)
}
Sim4 <- lapply(1:50, function(rep) {
  list(
    n50_1 = Sim4_data(50),
    n50_2 = Sim4_data(50),
    n400  = Sim4_data(400)
  )
})

#############################################
# (B) FIT THE FOUR MODELS ON EACH REPLICATION
#############################################

library(glmnet)

# 1) Helper: Naïve elastic net and corrected EN
naive.elastic.net <- function(X, y, lambda1, lambda2) {
  n <- nrow(X)
  p <- ncol(X)
  X_aug <- rbind(X, sqrt(lambda2) * diag(p)) / sqrt(1 + lambda2)
  y_aug <- c(y, rep(0, p))
  n_aug <- n + p
  lambda_aug <- lambda1 / sqrt(1 + lambda2)
  ## use in package 'glmnet' scaled by a factor of (1/2n) 
  lambda_glmnet <- lambda_aug * (2 * n_aug)
  fit <- glmnet(X_aug, y_aug, alpha = 1, lambda = lambda_glmnet,
                intercept = FALSE, standardize = FALSE)
  beta_hat_aug <- as.vector(coef(fit, s = lambda_glmnet))[-1]
  beta_naive <- beta_hat_aug / sqrt(1 + lambda2)
  return(beta_naive)
}

elastic.net <- function(X, y, lambda1, lambda2) {
  beta_naive <- naive.elastic.net(X, y, lambda1, lambda2)
  beta_en <- (1 + lambda2) * beta_naive
  return(beta_en)
}

# 2) Function to fit the 4 models using train/valid for tuning, then measure MSE on test
fit_four_models <- function(X_train, y_train, X_valid, y_valid, X_test, y_test) {
  # Standardize X based on training set
  train_means <- apply(X_train, 2, mean)
  train_sds   <- apply(X_train, 2, sd)
  
  X_train_std <- scale(X_train, center = train_means, scale = train_sds)
  X_valid_std <- scale(X_valid, center = train_means, scale = train_sds)
  X_test_std  <- scale(X_test,  center = train_means, scale = train_sds)
  
  # Center y
  y_mean <- mean(y_train)
  y_train_ctr <- y_train - y_mean
  y_valid_ctr <- y_valid - y_mean
  y_test_ctr  <- y_test  - y_mean
  
  out <- numeric(4)
  names(out) <- c("lasso", "ridge", "naive_en", "elastic_net")
  
  # LASSO
  lambda_grid <- 10^seq(-3, 3, length = 50)
  best_mse <- Inf
  best_lambda <- NULL
  for (lam in lambda_grid) {
    fit_lasso <- glmnet(X_train_std, y_train_ctr, alpha = 1, lambda = lam,
                        intercept = FALSE, standardize = FALSE)
    pred_valid <- predict(fit_lasso, X_valid_std, s = lam)
    mse_valid <- mean((y_valid_ctr - pred_valid)^2)
    if (mse_valid < best_mse) {
      best_mse <- mse_valid
      best_lambda <- lam
    }
  }
  fit_lasso_final <- glmnet(X_train_std, y_train_ctr, alpha = 1,
                            lambda = best_lambda, intercept = FALSE, standardize = FALSE)
  pred_test <- predict(fit_lasso_final, X_test_std, s = best_lambda)
  out["lasso"] <- mean((y_test_ctr - pred_test)^2)
  
  # RIDGE
  best_mse <- Inf
  best_lambda <- NULL
  for (lam in lambda_grid) {
    fit_ridge <- glmnet(X_train_std, y_train_ctr, alpha = 0, lambda = lam,
                        intercept = FALSE, standardize = FALSE)
    pred_valid <- predict(fit_ridge, X_valid_std, s = lam)
    mse_valid <- mean((y_valid_ctr - pred_valid)^2)
    if (mse_valid < best_mse) {
      best_mse <- mse_valid
      best_lambda <- lam
    }
  }
  fit_ridge_final <- glmnet(X_train_std, y_train_ctr, alpha = 0,
                            lambda = best_lambda, intercept = FALSE, standardize = FALSE)
  pred_test <- predict(fit_ridge_final, X_test_std, s = best_lambda)
  out["ridge"] <- mean((y_test_ctr - pred_test)^2)
  
  # Naïve Elastic Net
  lambda2_grid <- c(0.01, 0.1, 1, 10, 100)
  lambda1_grid <- 10^seq(-3, 3, length = 20)
  best_mse <- Inf
  best_pair <- c(NA, NA)
  for (l2 in lambda2_grid) {
    for (l1 in lambda1_grid) {
      beta_temp <- naive.elastic.net(X_train_std, y_train_ctr, l1, l2)
      pred_valid <- X_valid_std %*% beta_temp
      mse_valid <- mean((y_valid_ctr - pred_valid)^2)
      if (mse_valid < best_mse) {
        best_mse <- mse_valid
        best_pair <- c(l1, l2)
      }
    }
  }
  beta_naive_best <- naive.elastic.net(X_train_std, y_train_ctr, best_pair[1], best_pair[2])
  pred_test <- X_test_std %*% beta_naive_best
  out["naive_en"] <- mean((y_test_ctr - pred_test)^2)
  
  # Corrected Elastic Net
  best_mse <- Inf
  best_pair <- c(NA, NA)
  for (l2 in lambda2_grid) {
    for (l1 in lambda1_grid) {
      beta_temp <- elastic.net(X_train_std, y_train_ctr, l1, l2)
      pred_valid <- X_valid_std %*% beta_temp
      mse_valid <- mean((y_valid_ctr - pred_valid)^2)
      if (mse_valid < best_mse) {
        best_mse <- mse_valid
        best_pair <- c(l1, l2)
      }
    }
  }
  beta_en_best <- elastic.net(X_train_std, y_train_ctr, best_pair[1], best_pair[2])
  pred_test <- X_test_std %*% beta_en_best
  out["elastic_net"] <- mean((y_test_ctr - pred_test)^2)
  
  return(out)
}

#############################################
# (C) LOOP OVER 50 REPLICATES FOR EACH EXAMPLE
#     AND STORE TEST MSEs IN res1, res2, res3, res4
#############################################

# 1) Example 1
res1 <- list(
  mse_lasso = numeric(50),
  mse_en    = numeric(50),
  mse_ridge = numeric(50),
  mse_naive = numeric(50)
)
for (r in 1:50) {
  # train = n20_1, valid = n20_2, test = n200
  train_df <- Sim1[[r]]$n20_1
  valid_df <- Sim1[[r]]$n20_2
  test_df  <- Sim1[[r]]$n200
  
  X_train <- as.matrix(train_df[,-1])
  y_train <- train_df[,1]
  X_valid <- as.matrix(valid_df[,-1])
  y_valid <- valid_df[,1]
  X_test  <- as.matrix(test_df[,-1])
  y_test  <- test_df[,1]
  
  out <- fit_four_models(X_train, y_train, X_valid, y_valid, X_test, y_test)
  res1$mse_lasso[r] <- out["lasso"]
  res1$mse_en[r]    <- out["elastic_net"]
  res1$mse_ridge[r] <- out["ridge"]
  res1$mse_naive[r] <- out["naive_en"]
}

# 2) Example 2
res2 <- list(
  mse_lasso = numeric(50),
  mse_en    = numeric(50),
  mse_ridge = numeric(50),
  mse_naive = numeric(50)
)
for (r in 1:50) {
  train_df <- Sim2[[r]]$n20_1
  valid_df <- Sim2[[r]]$n20_2
  test_df  <- Sim2[[r]]$n200
  
  X_train <- as.matrix(train_df[,-1])
  y_train <- train_df[,1]
  X_valid <- as.matrix(valid_df[,-1])
  y_valid <- valid_df[,1]
  X_test  <- as.matrix(test_df[,-1])
  y_test  <- test_df[,1]
  
  out <- fit_four_models(X_train, y_train, X_valid, y_valid, X_test, y_test)
  res2$mse_lasso[r] <- out["lasso"]
  res2$mse_en[r]    <- out["elastic_net"]
  res2$mse_ridge[r] <- out["ridge"]
  res2$mse_naive[r] <- out["naive_en"]
}

# 3) Example 3
res3 <- list(
  mse_lasso = numeric(50),
  mse_en    = numeric(50),
  mse_ridge = numeric(50),
  mse_naive = numeric(50)
)
for (r in 1:50) {
  train_df <- Sim3[[r]]$n100_1
  valid_df <- Sim3[[r]]$n100_2
  test_df  <- Sim3[[r]]$n400
  
  X_train <- as.matrix(train_df[,-1])
  y_train <- train_df[,1]
  X_valid <- as.matrix(valid_df[,-1])
  y_valid <- valid_df[,1]
  X_test  <- as.matrix(test_df[,-1])
  y_test  <- test_df[,1]
  
  out <- fit_four_models(X_train, y_train, X_valid, y_valid, X_test, y_test)
  res3$mse_lasso[r] <- out["lasso"]
  res3$mse_en[r]    <- out["elastic_net"]
  res3$mse_ridge[r] <- out["ridge"]
  res3$mse_naive[r] <- out["naive_en"]
}

# 4) Example 4
res4 <- list(
  mse_lasso = numeric(50),
  mse_en    = numeric(50),
  mse_ridge = numeric(50),
  mse_naive = numeric(50)
)
for (r in 1:50) {
  train_df <- Sim4[[r]]$n50_1
  valid_df <- Sim4[[r]]$n50_2
  test_df  <- Sim4[[r]]$n400
  
  X_train <- as.matrix(train_df[,-1])
  y_train <- train_df[,1]
  X_valid <- as.matrix(valid_df[,-1])
  y_valid <- valid_df[,1]
  X_test  <- as.matrix(test_df[,-1])
  y_test  <- test_df[,1]
  
  out <- fit_four_models(X_train, y_train, X_valid, y_valid, X_test, y_test)
  res4$mse_lasso[r] <- out["lasso"]
  res4$mse_en[r]    <- out["elastic_net"]
  res4$mse_ridge[r] <- out["ridge"]
  res4$mse_naive[r] <- out["naive_en"]
}

#############################################
# (D) PRODUCE TABLE 2 AND FIGURE 4
#############################################

# Helper: bootstrap standard error of the median
bootstrap_se <- function(x, B = 500) {
  medians <- replicate(B, median(sample(x, replace = TRUE)))
  return(sd(medians))
}

# Finally, create the 2x2 boxplots (Figure 4)
par(mfrow = c(2, 2),   # 2x2 layout
    mar = c(4, 4, 2, 1)  # bottom, left, top, right margins
)

boxplot(res1$mse_lasso, res1$mse_en, res1$mse_ridge, res1$mse_naive,
        names = c("Lasso", "Enet", "Ridge", "NEnet"),
        main = "Example 1", ylab = "MSE")

boxplot(res2$mse_lasso, res2$mse_en, res2$mse_ridge, res2$mse_naive,
        names = c("Lasso", "Enet", "Ridge", "NEnet"),
        main = "Example 2", ylab = "MSE")

boxplot(res3$mse_lasso, res3$mse_en, res3$mse_ridge, res3$mse_naive,
        names = c("Lasso", "Enet", "Ridge", "NEnet"),
        main = "Example 3", ylab = "MSE")

boxplot(res4$mse_lasso, res4$mse_en, res4$mse_ridge, res4$mse_naive,
        names = c("Lasso", "Enet", "Ridge", "NEnet"),
        main = "Example 4", ylab = "MSE")

############################################################
# 2) A new function that produces a "wide" table, 
#    directly in the format of the paper
############################################################
create_wide_table <- function(res1, res2, res3, res4) {
  # We have four methods in this order:
  methods <- c("Lasso", "Elastic net", "Ridge regression", "Naïve elastic net")
  # We'll label the columns "Example 1", "Example 2", etc.
  ex_cols <- c("Example 1", "Example 2", "Example 3", "Example 4")
  
  # A small helper that returns "Median (SE)" as a string
  get_median_se <- function(x) {
    med <- round(median(x), 2)
    se  <- round(bootstrap_se(x), 2)
    paste0(med, " (", se, ")")
  }
  
  # We'll store results in a 4x4 matrix: rows=methods, cols=examples
  mat <- matrix("", nrow = 4, ncol = 4)
  
  # Fill row 1 (Lasso) for each example
  mat[1,1] <- get_median_se(res1$mse_lasso)
  mat[1,2] <- get_median_se(res2$mse_lasso)
  mat[1,3] <- get_median_se(res3$mse_lasso)
  mat[1,4] <- get_median_se(res4$mse_lasso)
  
  # Row 2 (Elastic net)
  mat[2,1] <- get_median_se(res1$mse_en)
  mat[2,2] <- get_median_se(res2$mse_en)
  mat[2,3] <- get_median_se(res3$mse_en)
  mat[2,4] <- get_median_se(res4$mse_en)
  
  # Row 3 (Ridge regression)
  mat[3,1] <- get_median_se(res1$mse_ridge)
  mat[3,2] <- get_median_se(res2$mse_ridge)
  mat[3,3] <- get_median_se(res3$mse_ridge)
  mat[3,4] <- get_median_se(res4$mse_ridge)
  
  # Row 4 (Naïve elastic net)
  mat[4,1] <- get_median_se(res1$mse_naive)
  mat[4,2] <- get_median_se(res2$mse_naive)
  mat[4,3] <- get_median_se(res3$mse_naive)
  mat[4,4] <- get_median_se(res4$mse_naive)
  
  # Convert to data frame, add "Method" column
  df <- data.frame(Method = methods, mat, stringsAsFactors = FALSE)
  colnames(df)[2:5] <- ex_cols
  return(df)
}

############################################################
# 3) Usage
############################################################
# After you have populated 'res1', 'res2', 'res3', 'res4'
# (each with $mse_lasso, $mse_en, $mse_ridge, $mse_naive),
# just do:
table2_wide <- create_wide_table(res1, res2, res3, res4)
print(table2_wide, row.names = FALSE)

# This prints a wide table with rows for each method and columns
# for "Example 1", "Example 2", etc. like:
#
#       Method               Example 1   Example 2   Example 3   Example 4
# 1     Lasso               3.06 (0.31) 3.87 (0.38) 65.0 (2.82) 46.6 (3.96)
# 2     Elastic net         2.51 (0.29) 3.16 (0.27) 56.6 (1.75) 34.5 (1.64)
# 3     Ridge regression    4.49 (0.46) 2.84 (0.27) 39.5 (1.80) 64.5 (4.78)
# 4     Naïve elastic net   5.70 (0.41) 2.73 (0.23) 41.0 (2.13) 45.9 (3.72)

