functions{
  real rad_angle(real angle){
    return(2*pi()*angle/360);
  }
}

data{
  int<lower=0> N;
  int<lower=0> NB;
  int<lower=0> NP;
  vector[N] S;
  vector[N] A;
  array[N] int<lower=0> B;
  array[N] int<lower=0> P;
}

transformed data{
  vector[N] log_S;
  log_S = log(130-S);
}

parameters{
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
  for(i in 1:N){
    mu_a[i] = b0 + b1[B[i]] + b2[P[i]];
  }
  for(i in 1:N){
    mu_s[i] = a0 + a1*cos(rad_angle(A[i]-10));
  }
}

model{
  sigma_s ~ exponential(1);
  sigma_a ~ exponential(1);
  b0 ~ normal(0,25);
  b1 ~ normal(0,1);
  b2 ~ normal(0,1);
  a0 ~ normal(10,1);
  a1 ~ normal(-5,1);
  A ~ normal(mu_a, sigma_a);
  log_S ~ normal(mu_s, sigma_s);
}








