##############################################################################################################
### make_moea_output_plots.py - python script to create plots for multi-objective optimization outputs
### Project started May 2017, last update Jan 2020
##############################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sbn
import importlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

### Project functions ###
import functions_moea_output_plots



sbn.set_style('white')
sbn.set_context('paper', font_scale=1.55)

eps = 1e-13
startTime = datetime.now()

dir_downloaded_inputs = './data/downloaded_inputs/'
dir_generated_inputs = './data/generated_inputs/'
dir_moea_output = './data/optimization_output/'
dir_figs = './figures/'





# ### get stochastic data
print('Reading in stochastic data..., ', datetime.now() - startTime)
synthetic_data = pd.read_csv(dir_generated_inputs + 'synthetic_data.txt', sep=' ')

### get historical simulation data
historical_simulation = pd.read_csv(dir_generated_inputs + 'historical_simulation.csv', index_col=0)

# ### constants
meanRevenue = np.mean(synthetic_data.revenue)
minSnowContract = 0.05
minMaxFund = 0.05
nYears=20
p_sfpuc = 150

### read in moea solutions for each LHC param sample
print('Reading in moea solutions..., ', datetime.now() - startTime)
importlib.reload(functions_moea_output_plots)
moea_solns_filtered = functions_moea_output_plots.get_moea_output(dir_generated_inputs, dir_moea_output, p_sfpuc,
                                                                  meanRevenue, minSnowContract, minMaxFund,
                                                                  debt_filter=True)
moea_solns_unfiltered = functions_moea_output_plots.get_moea_output(dir_generated_inputs, dir_moea_output, p_sfpuc,
                                                                    meanRevenue, minSnowContract, minMaxFund,
                                                                    debt_filter=False)

### choose 3 example policies for plotting from sfpuc baseline params
cases_sfpuc_index = [1585,1595,1597]
params_sfpuc = moea_solns_filtered.loc[moea_solns_filtered.p==p_sfpuc].iloc[0,:].loc[['Delta_debt','Delta_fund','c','delta','lam_prem_shift','expected_net_revenue']]



### plot Pareto front for sfpuc baseline parameters (fig 8)
# print('Plotting Pareto set for baseline parameters... (fig 8), ', datetime.now() - startTime)
# index of 3 cases to highlight in plot [A = high cash flow, B = compromise, C = low debt]
# functions_moea_output_plots.plot_pareto_baseline(dir_figs, moea_solns_filtered, p_sfpuc, cases_sfpuc_index)



### plot simulated state variables for 3 example policies over historical period
print('Plotting historical simulation for 3 policies, sfpuc baseline... (fig 8), ', datetime.now() - startTime)
functions_moea_output_plots.plot_example_simulations(dir_figs, moea_solns_filtered, params_sfpuc, cases_sfpuc_index, historical_simulation, meanRevenue)


# ### plot histogram of objectives for 3 policies for sfpuc baseline parameters (fig 9). Will also compare python objectives (validate) to c++ version (borg, retest) to validate monte carlo model
# print('Plotting histogram of objectives for 3 policies, sfpuc baseline (fig 9)..., ', datetime.now() - startTime)
# # index of 3 cases to highlight in plot [A = high cash flow, B = compromise, C = low debt]
# importlib.reload(functions_moea_output_plots)
# functions_moea_output_plots.plot_distribution_objectives(dir_figs, synthetic_data, moea_solns_filtered, cases_sfpuc_index, params_sfpuc, meanRevenue, nYears)



# ### plot tradeoff cloud of pareto fronts, filtered (fig 10)
# print('Plotting plot tradeoff cloud of pareto fronts, filtered (fig 10)..., ', datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# functions_moea_output_plots.plot_tradeoff_cloud(dir_figs, moea_solns_filtered, meanRevenue, p_sfpuc, debt_filter=True)

# ### plot tradeoff cloud of pareto fronts, unfiltered (fig S8)
# print('Plotting plot tradeoff cloud of pareto fronts, unfiltered (fig S8)..., ', datetime.now() - startTime)
# functions_moea_output_plots.plot_tradeoff_cloud(dir_figs, moea_solns_unfiltered, meanRevenue, p_sfpuc, debt_filter=False)



# ### plot sensitivity analysis for debt objective, filtered (fig 11)
# print('Plotting sensitivity analysis for debt objective, filtered (fig 11)..., ', datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# functions_moea_output_plots.plot_sensitivity_debt(dir_figs, moea_solns_filtered, p_sfpuc, debt_filter=True)

# ### plot sensitivity analysis for debt objective, unfiltered (fig S9)
# print('Plotting sensitivity analysis for debt objective, unfiltered (fig S9)..., ', datetime.now() - startTime)
# functions_moea_output_plots.plot_sensitivity_debt(dir_figs, moea_solns_unfiltered, p_sfpuc, debt_filter=False)



# ### plot sensitivity analysis for cash flow objective, filtered (fig 12)
# print('Plotting sensitivity analysis for cash flow objective, filtered (fig 12)..., ', datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# functions_moea_output_plots.plot_sensitivity_cashflow(dir_figs, moea_solns_filtered, p_sfpuc, meanRevenue, debt_filter=True)

# ### plot sensitivity analysis for cash flow objective, unfiltered (fig S10)
# print('Plotting sensitivity analysis for cash flow objective, unfiltered (fig S10)..., ', datetime.now() - startTime)
# functions_moea_output_plots.plot_sensitivity_cashflow(dir_figs, moea_solns_unfiltered, p_sfpuc, meanRevenue, debt_filter=False)



# ### get runtime metrics for moea runs, baseline & sensitivity params
# print('Getting runtime metrics for moea runs, baseline & sensitivity params..., ', datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# nSeedsBase = 50
# nSeedsSensitivity = 10
# nfe = 10000
# metrics_seedsBase, metrics_seedsSensitivity, p_successes = \
#   functions_moea_output_plots.get_metrics_all(dir_moea_output, p_sfpuc, nSeedsBase, nSeedsSensitivity)



# ### plot hypervolume for baseline (50 seeds) + sample of 12 sensitivity analysis runs (10 seeds) (fig S4)
# print('Plotting hypervolume (fig S4)..., ', datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# functions_moea_output_plots.plot_hypervolume(dir_figs, metrics_seedsBase, metrics_seedsSensitivity, p_successes, nSeedsBase, nSeedsSensitivity, nfe)

# ### plot generational distance for baseline (50 seeds) + sample of 12 sensitivity analysis runs (10 seeds) (fig S5)
# print('Plotting generational distance (fig S5)..., ', datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# functions_moea_output_plots.plot_generational_distance(dir_figs, metrics_seedsBase, metrics_seedsSensitivity, p_successes, nSeedsBase, nSeedsSensitivity, nfe)

# ### plot epsilon indicator for baseline (50 seeds) + sample of 12 sensitivity analysis runs (10 seeds) (fig S6)
# print('Plotting epsilon indicator (fig S6)..., ', datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# functions_moea_output_plots.plot_epsilon_indicator(dir_figs, metrics_seedsBase, metrics_seedsSensitivity, p_successes, nSeedsBase, nSeedsSensitivity, nfe)

# print('Finished, ', datetime.now() - startTime)






