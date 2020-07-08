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
import warnings
warnings.filterwarnings('ignore')

### Project functions ###
import functions_clean_data
import functions_synthetic_data
import functions_revenues_contracts

sbn.set_style('white')
sbn.set_context('paper', font_scale=1.55)

eps = 1e-13
startTime = datetime.now()

dir_downloaded_inputs = './data/downloaded_inputs/'
dir_generated_inputs = './data/generated_inputs/'
dir_figs = './figures/'



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
sweSynth = functions_synthetic_data.synthetic_swe(dir_generated_inputs, swe, redo = False, save = False)

# monthly generation, dependent on swe. Will also create fig S2, showing fitted models (gen as fn of swe) for each month.
print('Generating synthetic hydropower generation..., ', datetime.now() - startTime)
genSynth = functions_synthetic_data.synthetic_generation(dir_generated_inputs, dir_figs, gen, sweSynth,
                                                         redo = False, save = False, plot = False)


# monthly power price
print('Generating synthetic power prices..., ', datetime.now() - startTime)
importlib.reload(functions_synthetic_data)
powSynth = functions_synthetic_data.synthetic_power(dir_generated_inputs, power, redo = False, save = False)




### Simulate revenues and hedge payouts
# monthly revenues for SFPUC model
print('Generating simulated revenues..., ', datetime.now() - startTime)
importlib.reload(functions_revenues_contracts)
revHist, powHistSample, revSim = functions_revenues_contracts.simulate_revenue(dir_generated_inputs, gen, hp_GWh,
                                                                           hp_dolPerKwh, genSynth, powSynth,
                                                                           redo = False, save = False)


# get index from swe/revenue relationship.
nYr = int(len(revSim) / 12)
yrSim = np.full((1, nYr * 12), 0)
for i in range(1, nYr):
  yrSim[0, (12 * i):(12 * (i + 1))] = i
revSimWyr = revSim.groupby(yrSim[0, :(nYr * 12)]).sum()
revHistWyr = revHist.groupby('wyear').sum()
genSynthWyr = genSynth.groupby(yrSim[0, :(nYr * 12)]).sum()
genHistWyr = gen.groupby(yrSim[0, :len(powHistSample)]).sum()
powSynthWyr = powSynth.groupby(yrSim[0, :(nYr * 12)]).mean()
powHistWyr = powHistSample.groupby(yrSim[0, :len(powHistSample)]).mean()

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

# ### plot comparing historical vs synthetic for hydro generation (as function of wetness) and for power prices
# print('Plotting validation for hydropower generation and power price (fig 4)..., ', datetime.now() - startTime)
# functions_synthetic_data.plot_historical_synthetic_generation_power(dir_figs, gen, genSynth, power, powSynth)

# ### plots for SWE Feb vs Apr, Swe index vs Generation, & Swe index vs revenues
# importlib.reload(functions_revenues_contracts)
# print('Plotting validation for SWE Feb vs Apr, Swe index vs Generation, & Swe index vs revenues (fig 3)..., ', datetime.now() - startTime)
# functions_revenues_contracts.plot_SweFebApr_SweGen_SweRev(dir_figs, swe, gen, revHist, sweSynth, genSynth, revSim,
#                                                           sweWtParams, meanRevenue, fixedCostFraction, histRev = True)



# payout for swe-based capped contract for differences (cfd), centered around 50th percentile
# importlib.reload(functions_revenues_contracts)
print('Generating simulated CFD net payouts..., ', datetime.now() - startTime)
payoutPutSim = functions_revenues_contracts.snow_contract_payout(dir_generated_inputs, sweWtSynth, contractType='put',
                                                               lambdaRisk=0.25, strikeQuantile=0.5,
                                                               redo=False, save = False)

payoutShortCallSim = functions_revenues_contracts.snow_contract_payout(dir_generated_inputs, sweWtSynth,
                                                                     contractType='shortcall', lambdaRisk=0.25,
                                                                     strikeQuantile=0.5, redo=False, save = False)

payoutCfdSim = functions_revenues_contracts.snow_contract_payout(dir_generated_inputs, sweWtSynth, contractType = 'cfd',
                                                               lambdaRisk = 0.25, strikeQuantile = 0.5,
                                                               capQuantile = 0.95, redo = False, save = False)

# payoutFebSim = functions_revenues_contracts.snow_contract_payout(dir_generated_inputs, sweSynth.danFeb, contractType = 'cfd',
#                                                                lambdaRisk = 0.25, strikeQuantile = 0.5,
#                                                                capQuantile = 0.95, redo = True, save = True)
# payoutAprSim = functions_revenues_contracts.snow_contract_payout(dir_generated_inputs, sweSynth.danApr, contractType = 'cfd',
#                                                                lambdaRisk = 0.25, strikeQuantile = 0.5,
#                                                                capQuantile = 0.95, redo = True, save = True)



# ### get shift in cfd net payout function based on lambda, relative to baseline payouts with lambda = 0.25. Note, we do put here, since sell_call side of swap uses lambda=0 and will just be a constant shift.
# # read in lambdas from LHC sample
# print('Generating CFD loading shifts for different lambda values in sensitivity analysis..., ', datetime.now() - startTime)
# # read in lambdas from LHC sample
# importlib.reload(functions_revenues_contracts)
# param_list = pd.read_csv(dir_generated_inputs + 'param_LHC_sample.txt', sep=' ',
#                          header=None, names=['c','delta','Delta_fund','Delta_debt','lam'])
# # get premium shift for each lambda in dataset
# lam_params = functions_revenues_contracts.snow_contract_params_lambda(dir_generated_inputs, sweSynth, param_list.lam.values, contractType = 'cfd',
#                                                                       strikeQuantile = 0.5, capQuantile=0.95)
# param_list['lam_capX_2'] = lam_params[:,0]
# param_list['lam_capX_1'] = lam_params[:,1]
# param_list['lam_capX_0'] = lam_params[:,2]
# param_list['lam_capY_2'] = lam_params[:,3]
# param_list['lam_capY_1'] = lam_params[:,4]
# param_list['lam_capY_0'] = lam_params[:,5]

# param_list.to_csv(dir_generated_inputs + 'param_LHC_sample_withLamPricing.txt', sep=' ', header=True, index=False)


# ### get historical swe, gen, power price, revenue, net revenue. Period of record for hydropower = WY 1988-2016
# historical_data = pd.DataFrame({'sweFeb': swe.loc[revHistWyr.index,:].danFeb, 'sweApr': swe.loc[revHistWyr.index,:].danApr})
# historical_data['gen'] = genHistWyr.tot.values/1000
# powHistWyr.index = revHistWyr.index
# historical_data['pow'] = powHistWyr
# historical_data['rev'] = revHistWyr.rev

# historical_data.index = np.arange(1988, 2017)
# historical_data.to_csv(dir_generated_inputs + 'historical_data.csv', sep=' ')



### plot CFD contract as composite of put contract and short capped call contract (Fig S3)
# importlib.reload(functions_revenues_contracts)
# print('Plotting CFD contract as composite of put contract and short capped call contract (Fig S3)..., ', datetime.now() - startTime)
functions_revenues_contracts.plot_contract(dir_figs, sweWtSynth, payoutPutSim, payoutShortCallSim, payoutCfdSim,
                                       lambda_shifts=[0., 0.5], plot_type='composite')


# ### plot CFD contract with different loadings (fig 5)
# print('Plotting CFD contract with different loadings (fig 5)..., ', datetime.now() - startTime)
functions_revenues_contracts.plot_contract(dir_figs, sweWtSynth, payoutPutSim, payoutShortCallSim, payoutCfdSim,
                                       lambda_shifts=[0., 0.5], plot_type='lambda')


# ### plot of hedged & unhedged revenues in swe bins (fig 7)
# functions_revenues_contracts.plot_swe_hedged_revenue(dir_figs, sweWtSynth, revSimWyr, payoutCfdSim, meanRevenue, fixedCostFraction)


# ### get stats for contracts without reserve, as function of slope (fig 6)
# functions_revenues_contracts.plot_cfd_slope_effect(dir_figs, sweWtSynth, revSimWyr, payoutCfdSim, meanRevenue, fixedCostFraction)



# ### save data to use as inputs to moea for the current study
# print('Saving synthetic data..., ', datetime.now() - startTime)
# functions_revenues_contracts.save_synthetic_data_moea(dir_generated_inputs, sweSynth, revSimWyr)

print('Finished, ', datetime.now() - startTime)



