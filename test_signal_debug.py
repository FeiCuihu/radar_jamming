import numpy as np
from signal_radar_jamming import RadarJammingGenerator
import matplotlib.pyplot as plt
import os
import json
import csv
from datetime import datetime
import pywt
from scipy import io
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import re
import sqlite3
import uuid
from signal_radar_jamming import *

if __name__ == "__main__":

    signal = RadarJammingGenerator()

    lfm = signal.radar_signal
    if np.all(lfm == 0):
        print("lfm 全是零")
    else:
        print("第一种情况没问题")

    radar_params = {'f0': 1000000.0,
                    'Kam': 99999999999.99998,  # 调制斜率 (Hz/s)
                    'pulse_width': 4e-05,
                    'PRI': 5e-05,
                    'amplitude': 1.0
                    }
    signal.update_radar_signal(radar_params)

    lfm = signal.radar_signal

    if np.all(lfm == 0):
        print("lfm 全是零")
    else:
        print("第二种情况没问题")
