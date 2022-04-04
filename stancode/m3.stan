functions{
  real rad_angle(real angle){
    return(2*pi()*angle/360);
  }
  vector merge_missing(int[] miss_indexes, vector x_obs, vector x_miss){
    int N = dims(x_obs)[1];
    int N_miss = dims(x_miss)[1];
    vector[N] merged;
    merged = x_obs;
    for(i in 1:N_miss) merged[miss_indexes[i]] = x_miss[i];
    return merged;
  }
}

data{
  int<lower=0> N;
  int<lower=0> NB;
  int<lower=0> NP;
  int<lower=0> S_miss;
  array[S_miss] int<lower=0> S_missidx;
  vector[N] S;
  vector[N] A;
  array[N] int<lower=0> B;
  array[N] int<lower=0> P;
}

parameters{
  real<lower=0> sigma_s;
  real<lower=0> sigma_a;
  real a0;
  real a1;
  real b0;
  vector[NB] b1;
  vector[NP] b2;
  vector[S_miss] S_impute;
}

transformed parameters{
  vector[N] mu_s;
  vector[N] mu_a;
  vector[N] S_merge;
  vector[N] log_S_merge;
  for(i in 1:N){
    mu_a[i] = b0 + b1[B[i]] + b2[P[i]];
  }
  for(i in 1:N){
    mu_s[i] = a0 + a1*cos(rad_angle(A[i]-10));
  }
  S_merge = merge_missing(S_missidx, to_vector(S), S_impute);
  log_S_merge = log(130 - S_merge);
}

model{
  sigma_s ~ exponential(1);
  sigma_a ~ exponential(1);
  b0 ~ normal(0,25);
  b1 ~ normal(0,10);
  b2 ~ normal(0,10);
  a0 ~ normal(10,1);
  a1 ~ normal(-5,1);
  A ~ normal(mu_a, sigma_a);
  log_S_merge ~ normal(mu_s, sigma_s);
}











