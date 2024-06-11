
# function to apply to bin edges in output from np.histogram(...) to center them
#   INPUT: list or array containing bin edges
#   OUTPUT: list or array with center position of the bins
def center_bins_hist(bin_edges):
    return (bin_edges[1:] + bin_edges[:-1]) / 2