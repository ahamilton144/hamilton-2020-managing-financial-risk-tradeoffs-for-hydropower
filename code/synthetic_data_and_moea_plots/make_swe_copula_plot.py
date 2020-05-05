##############################################################################################################
### make_sswe_copula_plot.py - python script to create copula goodness of fit plot for SWE data
### Project started May 2017, last update Jan 2020
##############################################################################################################

import seaborn as sbn
import importlib
from datetime import datetime


### Project functions ###
import functions_clean_data
import functions_synthetic_data
import functions_revenues_contracts

sbn.set_style('white')
sbn.set_context('paper', font_scale=1.55)

eps = 1e-13
startTime = datetime.now()

dir_downloaded_inputs = '../../data/downloaded_inputs/'
dir_generated_inputs = '../../data/generated_inputs/'
dir_figs = '../../figures/'



### Get and clean data
print('Reading in data..., ', datetime.now() - startTime)
# SWE
importlib.reload(functions_clean_data)
swe = functions_clean_data.get_clean_swe(dir_downloaded_inputs)

### Plot empirical vs synthetic swe copula (Fig S1) - note: this function is slow
print('Plotting empirical vs synthetic swe copula (Fig S1)..., ', datetime.now() - startTime)
functions_synthetic_data.plot_empirical_synthetic_copula_swe(dir_figs, swe, startTime)

print('Finished..., ', datetime.now() - startTime)
