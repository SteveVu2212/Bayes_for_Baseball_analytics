functions{
  real rad_angle(real angle){
    return(2*pi()*angle/360);
  }
}

data{
  int<lower=0> N;
  int<lower=0> NB;
  int<lower=0> NP;
  real Ds;
  real Da;
  vector[N] S_obs;
  vector[N] A_obs;
  array[N] int<lower=0> B;
  array[N] int<lower=0> P;
}

parameters{
  vector[N] S_true;
  vector[N] A_true;
  real<lower=0> sigma_s;
  real<lower=0> sigma_a;
  real a0;
  real a1;
  real b0;
  vector[NB] b1;
  vector[NP] b2;
}

transformed parameters{
  vector[N] mu_s;
  vector[N] mu_a;
  vector[N] log_S_true;
  for(i in 1:N){
    mu_a[i] = b0 + b1[B[i]] + b2[P[i]];
  }
  for(i in 1:N){
    mu_s[i] = a0 + a1*cos(rad_angle(A_true[i]-10));
  }
  log_S_true = log(130 - S_true);
}

model{
  sigma_s ~ exponential(1);
  sigma_a ~ exponential(1);
  b0 ~ normal(0,25);
  b1 ~ normal(0,10);
  b2 ~ normal(0,10);
  a0 ~ normal(10,1);
  a1 ~ normal(-5,1);
  A_obs ~ normal(A_true, Da);
  A_true ~ normal(mu_a, sigma_a);
  S_obs ~ normal(S_true, Ds);
  log_S_true ~ normal(mu_s, sigma_s);
}








