import numpy
import pandas as pd
from numpy import genfromtxt

import csv, operator

fields = ['Date','Speeed','Angle']


csvDataDirectory = './Data_CSV/Saved_data.csv'

with open(csvDataDirectory) as csvfile:
    reader = csv.DictReader(csvfile,  delimiter = ";")
    for row in reader:
        angles =  (row['Angle'])

