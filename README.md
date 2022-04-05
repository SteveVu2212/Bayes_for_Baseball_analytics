# Authors

Steve Vu

# Introduction

The project aims at evaluating the performance of hitters in Major League Baseball (MLB). In 
the data set, each batter and pitcher are assigned unique ID. There are two systems, A and B, 
tracking two main metrics of exit velocity (EV) and launch angle (LA). There are measurement 
errors and missing data points from both system

By definition, exit velocity measures the speed of the baseball as it comes off the bat, 
immediately after a batter makes contact. Launch angle represents the vertical angle at which 
the ball leaves a player's bat after being struck. There are 5 hit types which are defined by 
the associated launch angle

The ultimate goal of the project is to predict the average exit velocity for each batter in the 
following season. The directed acyclic graph (DAG) is still employed to explicitly state the
assumptions and logic behind the model, which is essential to any further improvement

The project utilizes the Hamiltonian Monte Carlo (HMC) engine for Bayesian sampling and all models
are written in [Stan](https://mc-stan.org), a probabilistic programming language