##############################################################################################################
### make_synthetic_data_plots.py - python script to create synthetic data and related plots
### Project started May 2017, last update Jan 2020
##############################################################################################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sbn
import importlib
from datetime import datetime
import matplotlib.pyplot as plt


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

# hydro generation (GWh/mnth)
gen = functions_clean_data.get_historical_generation(dir_downloaded_inputs, swe).reset_index()

# wholesale power price ($/MWh), inflation adjusted
power = functions_clean_data.get_historical_power(dir_downloaded_inputs)

# SFPUC fin year sales and rates
hp_GWh, hp_dolPerKwh, hp_dolM = functions_clean_data.get_historical_SFPUC_sales()




### Generate synthetic time series
# # SWE, Feb 1 & Apr 1
print('Generating synthetic swe..., ', datetime.now() - startTime)
importlib.reload(functions_synthetic_data)
sweSynth = functions_synthetic_data.synthetic_swe(dir_generated_inputs, swe, redo = True, save = False)

# monthly generation, dependent on swe. Will also create fig S2, showing fitted models (gen as fn of swe) for each month.
print('Generating synthetic hydropower generation..., ', datetime.now() - startTime)
genSynth = functions_synthetic_data.synthetic_generation(dir_generated_inputs, dir_figs, gen, sweSynth,
                                                         redo = True, save = False)


# monthly power price
print('Generating synthetic power prices..., ', datetime.now() - startTime)
importlib.reload(functions_synthetic_data)
powSynth = functions_synthetic_data.synthetic_power(dir_generated_inputs, power, redo = True, save = False)




### Simulate revenues and hedge payouts
# monthly revenues for SFPUC model
print('Generating simulated revenues..., ', datetime.now() - startTime)
importlib.reload(functions_revenues_contracts)
revHist, revSim = functions_revenues_contracts.simulate_revenue(dir_generated_inputs, gen, hp_GWh,
                                                                           hp_dolPerKwh, genSynth, powSynth,
                                                                           redo = True, save = False)


# get index from swe/revenue relationship.
nYr = int(len(revSim) / 12)
yrSim = np.full((1, nYr * 12), 0)
for i in range(1, nYr):
  yrSim[0, (12 * i):(12 * (i + 1))] = i
revSimWyr = revSim.groupby(yrSim[0, :(nYr * 12)]).sum()
genSynthWyr = genSynth.groupby(yrSim[0, :(nYr * 12)]).sum()

lmRevSWE = sm.ols(formula='rev ~ sweFeb + sweApr', data=pd.DataFrame(
  {'rev': revSimWyr.values, 'sweFeb': sweSynth.danFeb.values,
   'sweApr': sweSynth.danApr.values}))
lmRevSWE = lmRevSWE.fit()
# print(lmRevSWE.summary())

sweWtParams = [lmRevSWE.params[1]/(lmRevSWE.params[1]+lmRevSWE.params[2]), lmRevSWE.params[2]/(lmRevSWE.params[1]+lmRevSWE.params[2])]
sweWtSynth = (sweWtParams[0] * sweSynth.danFeb + sweWtParams[1] * sweSynth.danApr)

genSynth['sweWt'] = (sweWtParams[0] * genSynth.sweFeb + sweWtParams[1] * genSynth.sweApr)
gen['sweWt'] = (sweWtParams[0] * gen.sweFeb + sweWtParams[1] * gen.sweApr)


### fixed cost parameters
meanRevenue = np.mean(revSimWyr)
fixedCostFraction =  0.914


### plot comparing historical vs synthetic for hydro generation (as function of wetness) and for power prices
print('Plotting validation for hydropower generation and power price (fig 4)..., ', datetime.now() - startTime)
functions_synthetic_data.plot_historical_synthetic_generation_power(dir_figs, gen, genSynth, power, powSynth,
                                                                    genOnly=False, genCombined=False, powerOnly=False)

### plots for SWE Feb vs Apr, Swe index vs Generation, & Swe index vs revenues
importlib.reload(functions_revenues_contracts)
print('Plotting validation for SWE Feb vs Apr, Swe index vs Generation, & Swe index vs revenues (fig 3)..., ', datetime.now() - startTime)
functions_revenues_contracts.plot_SweFebApr_SweGen_SweRev(dir_figs, swe, gen, revHist, sweSynth, genSynth, revSim,
                                                 sweWtParams, meanRevenue, fixedCostFraction, histRev = True)



# payout for swe-based capped contract for differences (cfd), centered around 50th percentile
importlib.reload(functions_revenues_contracts)
print('Generating simulated CFD net payouts..., ', datetime.now() - startTime)
payoutCfdSim = functions_revenues_contracts.snow_contract_payout(dir_generated_inputs, sweWtSynth, contractType = 'cfd',
                                                               lambdaRisk = 0.25, strikeQuantile = 0.5,
                                                               capQuantile = 0.95, redo = True, save = False)


# ### get shift in cfd net payout function based on lambda, relative to baseline payouts with lambda = 0.25. Note, we do put here, since sell_call side of swap uses lambda=0 and will just be a constant shift.
# read in lambdas from LHC sample
importlib.reload(functions_revenues_contracts)
print('Generating CFD loading shifts for different lambda values in sensitivity analysis..., ', datetime.now() - startTime)
param_list = pd.read_csv(dir_generated_inputs + 'param_LHC_sample.txt', sep=' ',
                         header=None, names=['c','delta','Delta_fund','Delta_debt','lam'])
# get premium shift for each lambda in dataset
lam_prem_shift = functions_revenues_contracts.snow_contract_payout_shift_lambda(sweWtSynth,
                                                                     param_list.iloc[:,-1].values, contractType = 'cfd',
                                                                     lambdaRisk = 0.25, strikeQuantile = 0.5)
param_list['lam_prem_shift'] = lam_prem_shift
param_list.to_csv(dir_generated_inputs + 'param_LHC_sample_withLamPremShift.txt', sep=' ', header=True, index=False)


### plot CFD contract as composite of put contract and short capped call contract (Fig S3)
importlib.reload(functions_revenues_contracts)
print('Plotting CFD contract as composite of put contract and short capped call contract (Fig S3)..., ', datetime.now() - startTime)
functions_revenues_contracts.plot_contract(dir_figs, sweWtSynth, payoutPutSim, payoutShortCallSim, payoutCfdSim,
                                       lambda_shifts=[0., 0.5], plot_type='composite')


### plot CFD contract with different loadings (fig 5)
print('Plotting CFD contract with different loadings (fig 5)..., ', datetime.now() - startTime)
functions_revenues_contracts.plot_contract(dir_figs, sweWtSynth, payoutPutSim, payoutShortCallSim, payoutCfdSim,
                                       lambda_shifts=[0., 0.5], plot_type='lambda')


### plot of hedged & unhedged revenues in swe bins (fig 7)
functions_revenues_contracts.plot_swe_hedged_revenue(dir_figs, sweWtSynth, revSimWyr, payoutCfdSim, meanRevenue, fixedCostFraction)


### get stats for contracts without reserve, as function of slope (fig 6)
functions_revenues_contracts.plot_cfd_slope_effect(dir_figs, sweWtSynth, revSimWyr, payoutCfdSim, meanRevenue, fixedCostFraction)



# ### save data to use as inputs to moea for the current study
print('Saving synthetic data..., ', datetime.now() - startTime)
functions_revenues_contracts.save_synthetic_data_moea(dir_generated_inputs, sweWtSynth, revSimWyr, payoutCfdSim)

print('Finished, ', datetime.now() - startTime)



# ### save monthly generation & power data for future study
print('Saving synthetic data..., ', datetime.now() - startTime)
importlib.reload(functions_revenues_contracts)
functions_revenues_contracts.save_synthetic_data_monthly(dir_generated_inputs, genSynth.gen, powSynth.powPrice)

print('Finished, ', datetime.now() - startTime)

