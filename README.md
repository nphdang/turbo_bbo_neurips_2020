# An improved version of Turbo algorithm for the Black-box optimization (BBO) competition organized by NeurIPS 2020
This implemention is our solution for the BBO competition organized by NeurIPS 2020 conference: https://bbochallenge.com/

# Final results
Our algorithm ranked the 4th place on the “warm start friendly” leaderboard: https://bbochallenge.com/altleaderboard 
It also ranked 17th place on the final leaderboard: https://bbochallenge.com/leaderboard

## Result on “warm start friendly” leaderboard
![leaderboard](https://github.com/nphdang/turbo_bbo_neurips_2020/blob/master/leaderboard.jpg)

# Main contributions
1. Reduce the bounds of hyper-parameters of each machine learning classifier
2. Improve the diversity in selecting batch elements
3. Increase the number of explored trust regions by using a base-length decay

# Installation
This implemention is based on Turbo algorithm. Please find the required packages from the Turbo's Github: https://github.com/uber-research/TuRBO

# How to run
This implemention is developed for optimizing machine learning models in the BBO competition. Please find the instruction from the competition Github: https://github.com/rdturnermtl/bbo_challenge_starter_kit/

# License
This implemention is released under Apache License 2.0 license except for the code derived from Turbo.
