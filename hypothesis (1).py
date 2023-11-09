import numpy as np
from random import shuffle
import json
from scipy.stats import norm
import sys

def permutationTest(x,y,n):
    #x - original vlaues
    #y- predicted values
    #n - number of permutations
    
    x = np.array(x)
    y = np.array(y)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    x = x.tolist()
    y = y.tolist()
    
    t_obs = abs(np.mean(x) - np.mean(y))
    d = x + y

    count = 0
    
    for i in range(n):
        d_i = d.copy()
        shuffle(d_i)
        x_i = d_i[:len(x)]
        y_i = d_i[len(x):]
        t_i = abs(np.mean(x_i) - np.mean(y_i))
        
        if t_i > t_obs:
            count += 1
    
    return count/n
def wald_test(list1, list2):
    mean1 = np.mean(list1)
    mean2 = np.mean(list2)
    var1 = np.var(list1, ddof=1)
    var2 = np.var(list2, ddof=1)
    n1 = len(list1)
    n2 = len(list2)
    
    # Calculate the test statistic
    test_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
    
    # Calculate the p-value using a standard normal distribution
    p_value = 2 * (1 - norm.cdf(np.abs(test_stat)))
    
    return test_stat, p_value
def wilcoxon_signed_rank(sample1, sample2):
    if len(sample1) != len(sample2):
        raise ValueError("The samples must have the same length.")

    differences = [sample1[i] - sample2[i] for i in range(len(sample1))]

    differences = [diff for diff in differences if diff != 0]

    ranked_diff = sorted(abs(diff) for diff in differences)

    ranks = {}
    for i, diff in enumerate(ranked_diff):
        if diff not in ranks:
            ranks[diff] = i + 1

    signed_ranks = [ranks[abs(diff)] if diff > 0 else -ranks[abs(diff)] for diff in differences]

    sum_positive_ranks = sum(rank for rank in signed_ranks if rank > 0)
    sum_negative_ranks = sum(rank for rank in signed_ranks if rank < 0)

    T = min(sum_positive_ranks, abs(sum_negative_ranks))

    n = len(signed_ranks)
    std_dev = (n * (n + 1) * (2 * n + 1)) / 6

    z = (T - 0.5) / std_dev

    p_value = 2 * (1 - abs(z))

    return T, p_value
if __name__ == "__main__":
    sample1 = sys.argv[1]
    sample2 = sys.argv[2]
    #here sample1 is the original cpcs values which are in json from extracted using extractsCPC.py
    #here sample2 is the predicted cpcs values which are in json from extracted using extractsCPC.py

    with open(sample1, 'r') as file:
        original_json = json.load(file)

    # Import predicted JSON file
    with open(sample2, 'r') as file:
        predicted_json = json.load(file)

    x = [value for value in original_json.values() if isinstance(value, (int, float))]
    y = [value for value in predicted_json.values() if isinstance(value, (int, float))]
    #permutation test
    statistic_perm = permutationTest(x, y, 1000)

    print("\n****************")
    print("Permutation Test") 
    print("****************\n")

    print("pValue: " + str(statistic_perm))

    if statistic_perm <= 0.05:
        print("Reject H0")
    else:
        print("Fail to Reject H0")
    
    
    #wald's test

    statistic_wald, p_value_wald = wald_test(x, y)
    print("\n****************")
    print("Wald's Test") 
    print("****************\n")
    print("Wald's test statistic:", statistic_wald)
    print("p-value:", p_value_wald)

    if statistic_wald <= 0.05:
        print("Reject H0")
    else:
        print("Fail to Reject H0")

    #wilcoxon test
    statistic_wilcoxon, p_value_wilcoxon = wilcoxon_signed_rank(x, y)

    # Print the test statistic and p-value
    print("\n****************")
    print("Wilcoxon Test") 
    print("****************\n")
    print("Wilcoxon Test Statistic:", statistic_wilcoxon)
    print("P-value:", p_value_wilcoxon)

    if p_value_wilcoxon <= 0.05:
        print("Reject H0")
    else:
        print("Fail to Reject H0")
