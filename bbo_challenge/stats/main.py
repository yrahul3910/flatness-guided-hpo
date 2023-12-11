from breast import results as breast_results
from digits import results as digits_results
from iris import results as iris_results
from wine import results as wine_results
from diabetes import results as diabetes_results
from experiment_analysis import run_stats
import sys


dataset = sys.argv[1]
results = {
    'breast': breast_results,
    'digits': digits_results,
    'iris': iris_results,
    'wine': wine_results,
    'diabetes': diabetes_results
}
run_stats(results[dataset])
