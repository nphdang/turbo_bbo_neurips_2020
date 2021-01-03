# An improved version of Turbo algorithm for the Black-box optimization (BBO) competition organized by NeurIPS 2020
This algorithm ranked the 4th place on the “warm start friendly” leaderboard of the BBO competition organized by NeurIPS 2020 conference: https://bbochallenge.com/altleaderboard

## Final result on “warm start friendly” leaderboard
![leaderboard](https://github.com/nphdang/turbo_bbo_neurips_2020/blob/master/leaderboard.jpg)

# Main contributions
1. Reduce the bounds of hyper-parameters of each machine learning classifer
2. Improve the diversity in selecting batch elements
3. Increase the number of explored trust regions by using a base-length decay

# Installation
This code is based on Turbo algorithm. Please find the required packages from Turbo's Github: https://github.com/uber-research/TuRBO

# How to run
This code is developed for optimizing machine learning models in the BBO competition. Please find the instruction from BBO competition Github: https://github.com/rdturnermtl/bbo_challenge_starter_kit/

# License
This code is released under FREE license except for the code derived from Turbo.
