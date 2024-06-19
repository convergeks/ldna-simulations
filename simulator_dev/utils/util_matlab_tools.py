import matlab
import numpy as np

# --- scripts ---
def util_conv_matlab_arr_to_list(mat_arr: matlab.double):
    return list(np.array(mat_arr).T[0])
    
