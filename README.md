# Strong convexity-based HPO

This repo contains the full source code and results for the strong convexity-based HPO paper.

* `image/` - Experiments on MNIST and SVHN
* `hpobench/` - Experiments on the OpenML datasets
* `bbo_challenge/` - Experiments on the Bayesmark datasets

In a newer commit, we have moved the results to an S3 bucket to save space. These are publicly available on `s3://stcvx-results/`. Because earlier commits had some results, we recommend that you clone using `git clone --depth 1` to avoid downloading the results.