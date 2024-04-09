import numpy as np
import os, sys

# -- Params --


# --- functions ---
def convert_dbm_watts(dBm:float = 0):
    mWatt = 10**(dBm/10)
    Watts = mWatt/1000
    return Watts