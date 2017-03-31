import numpy as np
import pandas as pd
from asl_data import AslDb


asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']