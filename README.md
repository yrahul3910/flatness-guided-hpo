# Flatness-guided HPO

This repo contains the full source code and results for the flatness-guided HPO paper. While this was in development,
the method was also called STCVX or AHSC temporarily, so you will likely find files with those names instead.

* `image/` - Experiments on MNIST and SVHN
* `hpobench/` - Experiments on the OpenML datasets
* `bbo_challenge/` - Experiments on the Bayesmark datasets

In a newer commit, we have moved the results to an S3 bucket to save space. These are publicly available on `s3://stcvx-results/`. Because earlier commits had some results, we recommend that you clone using `git clone --depth 1` to avoid downloading the results.
