##############################################################################################################
### functions_revenues_contracts.py - python functions used in creating simulated synthetic revenues and
###     index contract payouts, plus related plots
### Project started May 2017, last update Jan 2020
##############################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import statsmodels.formula.api as sm
import seaborn as sbn
from scipy import stats as st
from scipy.optimize import minimize


sbn.set_style('white')
sbn.set_context('paper', font_scale=1.55)

cmap = cm.get_cmap('viridis')
col = [cmap(0.1),cmap(0.3),cmap(0.6),cmap(0.8)]


N_SAMPLES = 1000000
eps = 1e-13



##########################################################################
######### Simulate revenue, matching SFPUC 2016 rates and demands ###########
############## Returns dataframe of monthly revenues ($M/mnth) #########################################
##########################################################################

def simulate_revenue(dir_generated_inputs, gen, hp_GWh, hp_dolPerKwh, genSynth, powSynth, redo = False, save = False):
  if (redo):
    # nYr = int(len(powSynth) / 12)
    # yrSim = np.full((1, nYr * 12), 0)
    # for i in range(1, nYr):
    #     yrSim[0, (12 * i):(12 * (i + 1))] = i

    ### rel b/w swe/gen and mtid
    # plt.scatter(hp_GWh.mtid.loc[2010:2016], swe.danWtAvg.loc[2010:2016])
    # np.corrcoef(hp_GWh.mtid.loc[2010:2016], swe.danWtAvg.loc[2010:2016])
    # get total amt above muni demand in each yr
    gen['aboveMuni'] = gen.tot - hp_GWh['M'].iloc[hp_GWh.shape[0] - 1] / 12
    gen['mtid'] = gen.aboveMuni.apply(lambda x: max(x, 0))
    # gen.mtid.loc[(gen.wmnth < 7) ] = 0     # assume mtid only buys power Apr-Sept
    hp_GWh['estMtid'] = np.nan
    hp_GWh.estMtid.loc[2010:2016] = gen.loc[gen.wyear > 2009, :].groupby('wyear').sum().mtid
    # plt.scatter(hp_GWh.estMtid.loc[2010:2016], hp_GWh.mtid.loc[2010:2016])
    # np.corrcoef(hp_GWh.estMtid.loc[2010:2016], hp_GWh.mtid.loc[2010:2016])
    # reg to get percentage estimated for mtid
    lmMtid = sm.ols(formula='mtid ~ est-1',
                    data=pd.DataFrame({'mtid': hp_GWh.mtid.loc[2010:2016], 'est': hp_GWh.estMtid.loc[2010:2016]}))
    lmMtid = lmMtid.fit()
    # print(lmMtid.summary())

    # plt.scatter(hp_GWh.estMtid.loc[2010:2016], lmMtid.predict())
    mtidGrowFrac = lmMtid.params[0]
    # gen.mtid = gen.mtid * mtidGrowFrac
    # gen['aboveMuniMtid'] = np.where(gen.aboveMuni > 0, gen.aboveMuni - gen.mtid, gen.aboveMuni)
    # plt.plot(gen.wmnth.loc[gen.wyear==2012],gen.aboveMuni.loc[gen.wyear==2012])
    # plt.plot(gen.aboveMuni)
    # plt.plot(gen.aboveMuniMtid)
    # plt.plot(gen.mtid)


    # revenue model: monthly gen & price, assume const demand to muni, 48% surplus (from regression above) to mtid throughout year (only if mtid rate < wholesale).
    # Rest to Wholesale. Also must buy power to meet unmet muni.
    def revenue_model_milDollars(sampGen_GWh, sampPow_DolPerkWh, dem_M_GWh, mtidFrac, rate_DolPerkWh_M,
                           rate_DolPerkWh_mtid):
      dem_mtid_GWh = np.maximum((sampGen_GWh - dem_M_GWh) * mtidFrac, 0)
      dem_mtid_GWh.loc[(sampPow_DolPerkWh < rate_DolPerkWh_mtid).values] = 0
      # dem_mtid_GWh.loc[(dem_mtid_GWh.index % 12 < 6)] = 0  # assume mtid only buys power Apr-Sept
      rev = (dem_M_GWh * rate_DolPerkWh_M + dem_mtid_GWh * rate_DolPerkWh_mtid + \
             (sampGen_GWh - dem_M_GWh - dem_mtid_GWh) * sampPow_DolPerkWh)
      return (rev)  # returns revenues in $Mil

    # simulated revs for synthetic time series
    revSim = revenue_model_milDollars(genSynth.gen,
                                powSynth.powPrice/1000,
                                hp_GWh['M'].iloc[hp_GWh.shape[0] - 1] / 12,
                                mtidGrowFrac,
                                hp_dolPerKwh['M'].iloc[hp_dolPerKwh.shape[0] - 1] ,
                                hp_dolPerKwh['mtid'].iloc[hp_dolPerKwh.shape[0] - 1] )
    powHistSample = powSynth.powPrice.iloc[3600:(3600+len(gen.tot))].reset_index(drop=True)
    # simulated revs for historical generation w/ random synth power price & current fixed muni/mtid rates
    revHist = pd.DataFrame({'rev': revenue_model_milDollars(gen.tot.reset_index(drop=True),
                                                      powSynth.powPrice.iloc[3600:(3600+len(gen.tot))].reset_index(drop=True)/1000,
                                                      hp_GWh['M'].iloc[hp_GWh.shape[0] - 1] / 12,
                                                      mtidGrowFrac,
                                                      hp_dolPerKwh['M'].iloc[hp_dolPerKwh.shape[0] - 1],
                                                      hp_dolPerKwh['mtid'].iloc[hp_dolPerKwh.shape[0] - 1]),
                            'wmnth': gen.wmnth,
                            'wyear': gen.wyear})

    if (save):
      revSim.to_pickle(dir_generated_inputs + 'revSim.pkl')
      revHist.to_pickle(dir_generated_inputs + 'revHist.pkl')
      powHistSample.to_pickle(dir_generated_inputs + 'powHistSample.pkl')


  else:
    revSim = pd.read_pickle(dir_generated_inputs + 'revSim.pkl')
    revHist = pd.read_pickle(dir_generated_inputs + 'revHist.pkl')
    powHistSample = pd.read_pickle(dir_generated_inputs + 'powHistSample.pkl')

  return (revHist, powHistSample, revSim)




##########################################################################
######### plot synthetic feb 1 vs apr 1 swe, sweWt vs gen, sweWt vs revenue (fig 3)###########
############## Returns figure #########################################
##########################################################################

def plot_SweFebApr_SweGen_SweRev(dir_figs, swe, gen, revHist, sweSynth, genSynth, revSim, sweWtParams,
                                MEAN_REVENUE, COST_FRACTION, histRev):
  nYr = int(len(revSim) / 12)
  yrSim = np.full((1, nYr * 12), 0)
  for i in range(1, nYr):
    yrSim[0, (12 * i):(12 * (i + 1))] = i
  revSimWyr = revSim.groupby(yrSim[0, :(nYr * 12)]).sum()
  revHistWyr = revHist.groupby('wyear').rev.sum()
  revSimWyr = revSimWyr - MEAN_REVENUE*COST_FRACTION
  revHistWyr = revHistWyr - MEAN_REVENUE*COST_FRACTION

  genSynthWyr = genSynth.groupby('wyr').sum()
  genWyr = gen.groupby('wyear').sum()

  sweWtSynth = (sweWtParams[0] * sweSynth.danFeb + sweWtParams[1] * sweSynth.danApr)
  sweWtHist = (sweWtParams[0] * swe.danFeb + sweWtParams[1] * swe.danApr)
  genSynthWyr['sweWt'] = (sweWtParams[0] * genSynthWyr['sweFeb'] + sweWtParams[1] * genSynthWyr['sweApr'])
  genWyr['sweWt'] = (sweWtParams[0] * genWyr['sweFeb'] + sweWtParams[1] * genWyr['sweApr'])

  plt.figure()
  ax = plt.subplot2grid((2, 3), (0, 0))
  ax.set_xlabel('Feb 1 SWE (inch)')
  ax.set_ylabel('Apr 1 SWE (inch)')
  ax.xaxis.set_label_position('top')
  ax.set_xticks(np.arange(0,76,25))
  ax.set_yticks(np.arange(0,76,25))
  ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=True)
  ax.scatter(sweSynth.danFeb.iloc[:500], sweSynth.danApr.iloc[:500], marker='o', facecolors='none',
              linewidth=1, alpha=0.7, edgecolors=col[3], s=30)
  ax.scatter(swe.danFeb, swe.danApr, color=col[0], alpha=0.6, marker='^', s=40)

  ax = plt.subplot2grid((2, 3), (1, 0))
  ax.set_xlabel('SWE Index (inch)')
  ax.set_ylabel('Generation (TWh/yr)')
  ax.set_xticks(np.arange(0,76,25))
  p1 = ax.scatter(sweWtSynth.iloc[:500], genSynthWyr.gen.iloc[:500]/1000, marker='o', facecolors='none',
              linewidth=1, alpha=0.7, edgecolors=col[3], s=30)
  p2 = ax.scatter(sweWtHist.loc[1988:], genWyr.tot/1000, color=col[0], alpha=0.6, marker='^', s=40)

  ax = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)
  ax.set_xlabel('SWE Index (inch)')
  ax.set_ylabel('Net Revenue ($M/year)')
  ax.yaxis.set_label_position('right')
  ax.set_xticks(np.arange(0,76,25))
  ax.tick_params(axis='y', which='both', labelleft=False, labelright=True)
  plt.axhline(0,ls=':',c='lightgrey')
  ax.scatter(sweWtSynth.iloc[:500], revSimWyr.iloc[:500], marker='o', facecolors='none',
              linewidth=1, alpha=0.7, edgecolors=col[3], s=30)
  if (histRev):
    ax.scatter(sweWtHist.loc[1988:], revHistWyr, alpha=0.6, color=col[0], marker='^', s=40)
  ax.legend([p1,p2], ['Synthetic', 'Historic'], loc='lower right')
  plt.savefig(dir_figs + 'fig3.jpg', bbox_inches='tight', dpi=1200)



##########################################################################
######### wang transform function ###########
############## Returns dataframe with net payout #########################################
##########################################################################

def wang(df, contractType, lam, k, cap=-1., premOnly=False, lastYrTrig=-1., count=0):  # df should be dataframe with columns 'asset' and 'prob'; contractType is 'put' or 'call'
  # print(count)
  if contractType == 'put':
    df['payout'] = df['asset'].apply(lambda x: max(k - x, 0))
    lam = -abs(lam)
  elif contractType == 'call':
    df['payout'] = df['asset'].apply(lambda x: max(x - k, 0))
    lam = abs(lam)
  elif contractType == 'shortcall':
    df['payout'] = df['asset'].apply(lambda x: max(-max(x - k, 0), -(cap - k)))
    lam = -abs(lam)
  elif contractType == 'putWithLastYrTrig':
    df['payout'] = df['asset'].apply(lambda x: max(k - x, 0))
    gttrig = (df['asset'].iloc[:-1] > lastYrTrig).values
    df['payout'].iloc[1:].loc[gttrig] = 0
    df['payout'].iloc[0] = 0
    lam = -abs(lam)
  else:
    df['payout'] = np.nan
  df.sort_values(inplace=True, by='payout')
  dum = df['prob'].cumsum()  # asset cdf
  dum = st.norm.cdf(st.norm.ppf(dum) + lam)  # risk transformed payout cdf
  dum = np.append(dum[0], np.diff(dum))  # risk transformed asset pdf
  prem = (dum * df['payout']).sum()
  if premOnly == False:
    df.sort_index(inplace=True)
    return ((df['payout'] - prem))
  else:
    return prem




##########################################################################
######### snow index contract net payouts ###########
############## Returns dataframe with net payout #########################################
##########################################################################

def snow_contract_payout(dir_generated_inputs, sweWtSynth, contractType = 'put', lambdaRisk = 0.25, strikeQuantile = 0.6,
                       capQuantile = 0.95, redo = False, save = False):

  if (redo):
    if (contractType == 'put'):
      snowPayoutSim = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': 1/sweWtSynth.shape[0]}), contractType='put',
                           lam=lambdaRisk, k=sweWtSynth.quantile(strikeQuantile), premOnly=False)
      if (save):
        save_location = dir_generated_inputs + 'payoutPut%sSim.pkl' % int(strikeQuantile*100)
        snowPayoutSim.to_pickle(save_location)

    if (contractType == 'shortcall'):
      snowPayoutSim = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': 1/sweWtSynth.shape[0]}), contractType='shortcall',
                           lam=lambdaRisk, k=sweWtSynth.quantile(strikeQuantile),
                           cap=sweWtSynth.quantile(capQuantile), premOnly=False)
      if (save):
        save_location = dir_generated_inputs + 'payoutShortCall%sSim.pkl' % int(strikeQuantile*100)
        snowPayoutSim.to_pickle(save_location)

    elif (contractType == 'cfd'):
      snowPayoutSim = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': 1/sweWtSynth.shape[0]}), contractType='put',
                           lam=lambdaRisk, k=sweWtSynth.quantile(strikeQuantile), premOnly=False)
      snowPayoutSim = snowPayoutSim + wang(pd.DataFrame({'asset': sweWtSynth, 'prob': 1 / sweWtSynth.shape[0]}),
                                           contractType='shortcall', lam=0, k=sweWtSynth.quantile(strikeQuantile),
                                           cap=sweWtSynth.quantile(capQuantile), premOnly=False)
      if (save):
        snowPayoutSim.to_pickle(dir_generated_inputs + 'payoutCfdSim.pkl')

  else:
    if (contractType == 'put'):
      save_location = dir_generated_inputs + 'payoutPut%sSim.pkl' % int(strikeQuantile * 100)
      snowPayoutSim = pd.read_pickle(save_location)
    if (contractType == 'shortcall'):
      save_location = dir_generated_inputs + 'payoutShortCall%sSim.pkl' % int(strikeQuantile * 100)
      snowPayoutSim = pd.read_pickle(save_location)
    elif (contractType == 'cfd'):
      snowPayoutSim = pd.read_pickle(dir_generated_inputs + 'payoutCfdSim.pkl')

  return (snowPayoutSim)



##########################################################################
######### get shift in cfd loading based on lambda, relative to baseline payouts with lambda = 0.25.
# Note, we do put here, since sell_call side of swap uses lambda=0 and will just be a constant shift. ###########
############## Returns dataframe with premium shift for each lambda #########################################
##########################################################################

def snow_contract_payout_shift_lambda(sweWtSynth, lam_list, contractType, lambdaRisk, strikeQuantile):
  if (contractType == 'cfd'):
    strike = sweWtSynth.quantile(strikeQuantile)
    prob = 1/sweWtSynth.shape[0]
    # first get prem for base case
    prem_base = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': prob}), contractType='put',
                     lam=lambdaRisk, k=strike, premOnly=True)
    # now get shift for every lambda in dataset
    lam_prem_shift = np.empty(lam_list.shape[0])
    for i in range(lam_list.shape[0]):
      lam_prem_shift[i] = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': prob}), contractType='put',
                               lam=lam_list[i], k=strike, premOnly=True, count=i) - prem_base

  return (lam_prem_shift)





##########################################################################
######### plot snow contract types (fig 5/S3) ###########
############## Returns figure #########################################
##########################################################################
def plot_contract(dir_figs, sweWtSynth, payoutPutSim, payoutShortCallSim, payoutCfdSim, lambda_shifts, plot_type):

  strike = sweWtSynth.quantile(0.5)
  prob = 1 / sweWtSynth.shape[0]
  # first get prem for base case
  prem_base = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': prob}), contractType='put', lam=0.25, k=strike, premOnly=True)
  for i in range(len(lambda_shifts)):
    lambda_shifts[i] = prem_base - wang(pd.DataFrame({'asset': sweWtSynth, 'prob': prob}), contractType='put',
                                        lam=lambda_shifts[i], k=strike, premOnly=True)

  ### plot regime as function of debt and uncertain params
  plt.figure()
  ax = plt.subplot2grid((3,1), (0, 0))
  # ax.set_xlabel('SWE Index (inch)')
  ax.set_ylabel('Density')
  # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
  ax.set_xlim([0,60])
  ax.set_ylim([0,0.04])
  # ax.set_yticks(np.arange(0, 6))
  ax.tick_params(axis='x', which='both', labelbottom=False,labeltop=True)
  ax.tick_params(axis='y', which='both', labelleft=False,labelright=False)
  # ax.xaxis.set_label_position('top')

  sbn.kdeplot(sweWtSynth, ax=ax, c='k', lw=2)

  ax = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
  ax.set_xlabel('SWE Index (inch)')
  ax.set_ylabel('Net Payout ($M)')
  ax.set_xlim([0, 60])
  ax.tick_params(axis='y', which='both', labelleft=False,labelright=False)
  ax.axhline(0, color='grey', linestyle=':')
  kinkY = np.min(payoutCfdSim)
  kinkX = np.min(sweWtSynth.loc[payoutCfdSim < kinkY + eps])
  line3, = ax.plot([0, kinkX, 60], [kinkX + kinkY, kinkY, kinkY], color=col[0], linewidth=2)
  if (plot_type == 'lambda'):
    line4, = ax.plot([0, kinkX, 60], [kinkX + kinkY + lambda_shifts[0], kinkY + lambda_shifts[0], kinkY + lambda_shifts[0]], color=col[0], ls='--', linewidth=2)
    line5, = ax.plot([0, kinkX, 60], [kinkX + kinkY + lambda_shifts[1], kinkY + lambda_shifts[1], kinkY + lambda_shifts[1]], color=col[0], ls=':', linewidth=2)
    plt.legend([line4,line3,line5],['No loading', 'Baseline loading', 'High loading'],loc='upper right')
    plot_name = dir_figs + 'fig5.jpg'
  elif (plot_type=='composite'):
    # plot put
    kinkY = np.min(payoutPutSim)
    kinkX = np.min(sweWtSynth.loc[payoutPutSim < kinkY + eps])
    line1, = ax.plot([0, kinkX, 60], [kinkX + kinkY, kinkY, kinkY], color=col[3],  lw=2, ls='--')
    # plot shortcall
    kinkStrikeY = np.max(payoutShortCallSim)
    kinkStrikeX = np.max(sweWtSynth.loc[payoutShortCallSim > kinkStrikeY - eps])
    kinkCapY = np.min(payoutShortCallSim)
    kinkCapX = np.min(sweWtSynth.loc[payoutShortCallSim < kinkCapY + eps])
    line2, = ax.plot([0, kinkStrikeX, kinkCapX, 60], [kinkStrikeY, kinkStrikeY, kinkCapY, kinkCapY], color=col[2], lw=2, ls='--')
    plt.legend([line1,line2,line3],['Long put','Short capped call','Capped contract for differences'],loc='lower left')
    plot_name = dir_figs + 'figS3.jpg'

  plt.savefig(plot_name, dpi=1200)

  return


##########################################################################
######### Get maximally-hedging contract value, based on risk quantile ####
### returns value ####
# ##########################################################################
def get_max_hedge(revSimWyr, payoutCfdSim, riskQuantile = 0.05, nSamplesOptimization = 10000):
  sample_years = np.random.choice(range(1, revSimWyr.shape[0]), size = nSamplesOptimization)
  revSimWyr_sample = revSimWyr.iloc[sample_years]
  payoutCfdSim_sample = payoutCfdSim.iloc[sample_years]

  def get_risk_quantile(x):
    return (-((revSimWyr_sample + x * payoutCfdSim_sample).quantile(riskQuantile)))

  value_contract = minimize(get_risk_quantile, x0 = 1., method = 'nelder-mead', options = {'xtol': 1e-5, 'disp': True})

  return (value_contract.x[0])


##########################################################################
######### plot index-revenue distribution w/ & w/o cfd (fig 7)###########
############## Returns figure #########################################
##########################################################################
def plot_swe_hedged_revenue(dir_figs, sweWtSynth, revSimWyr, payoutCfdSim, meanRevenue, fixedCostFraction):

  netRevSimWyr = revSimWyr - meanRevenue * fixedCostFraction
  # get contract weights by optimizing for VAR95
  slope_cfd = get_max_hedge(netRevSimWyr, payoutCfdSim, 0.05, 1000000)
  # slope_cfd =  0.9873901367187501

  plt_ylim = [-28,55]

  # plot as errorbars for bins
  def get_quantiles(rev, swe, sweBinSize):
    revQuants = pd.DataFrame({'sweBound': [1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
                              'meanRev': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]})['dum'].apply(
                                lambda x: np.where(x < 9, rev.loc[
                                  (swe >= (x-1)*sweBinSize) & (swe < x*sweBinSize)].mean(),
                                                   np.where(x < 10, rev.loc[
                                                     (swe >= (x-1)*sweBinSize)].mean(),
                                                            rev.mean()))),
                              'q5': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]})['dum'].apply(
                                lambda x: np.where(x < 9, rev.loc[
                                  (swe >= (x - 1) * sweBinSize) & (swe < x * sweBinSize)].quantile(0.05),
                                                   np.where(x < 10, rev.loc[
                                                     (swe >= (x-1)*sweBinSize)].quantile(0.05),
                                                            rev.quantile(0.05)))),
                              'q95': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]})['dum'].apply(
                                lambda x: np.where(x < 9, rev.loc[
                                  (swe >= (x - 1) * sweBinSize) & (swe < x * sweBinSize)].quantile(0.95),
                                                   np.where(x < 10, rev.loc[
                                                     (swe >= (x - 1) * sweBinSize)].quantile(0.95),
                                                            rev.quantile(0.95))))
                              })
    revQuants.sweBound = revQuants.sweBound * sweBinSize
    return revQuants


  sweBinSize = 8
  netRevSimWyrQuants = get_quantiles(netRevSimWyr, sweWtSynth, sweBinSize)
  netRevSimCfdQuants = get_quantiles(netRevSimWyr + slope_cfd * payoutCfdSim, sweWtSynth, sweBinSize)
  plotSpacers = [1.3, 0.7]

  plt.figure()
  plt.axhline(0,linestyle=':',c='grey')
  for i in range(10):
    plt.axvline(i * sweBinSize, color='lightgrey', lw=1)
  plt.axvline(10 * sweBinSize, color='black', lw=1.5)
  for i in range(11, 13):
    plt.axvline(i * sweBinSize, color='lightgrey', lw=1)
  plt.ylim(plt_ylim)
  plt.xlim([0, sweBinSize * 10])
  plt.xticks(np.arange(0, sweBinSize * 13, sweBinSize),
             [0, 8, 16, 24, 32, 40, 48, 56, 64, '$\infty$', ' ', '0', '$\infty$'])
  plt.tick_params(labeltop=False, labelright=True)
  plt.xlabel('SWE Index Bins (inch)')
  plt.ylabel('Net Revenue ($M/year)')
  eb1 = plt.errorbar(netRevSimWyrQuants.sweBound - sweBinSize / 2 * plotSpacers[0], netRevSimWyrQuants.meanRev,
                     yerr=pd.DataFrame({'errPlus': netRevSimWyrQuants.q95 - netRevSimWyrQuants.meanRev,
                                        'errMin': netRevSimWyrQuants.meanRev - netRevSimWyrQuants.q5}
                                       ).transpose().values,
                     marker='s', ms=7, mew=1, mec=col[3], color=col[3], linestyle='None')
  eb1[-1][0].set_linewidth(3)

  eb3 = plt.errorbar(netRevSimCfdQuants.sweBound - sweBinSize / 2 * plotSpacers[1], netRevSimCfdQuants.meanRev,
                     yerr=pd.DataFrame({'errPlus': netRevSimCfdQuants.q95 - netRevSimCfdQuants.meanRev,
                                        'errMin': netRevSimCfdQuants.meanRev - netRevSimCfdQuants.q5}
                                       ).transpose().values,
                     marker='^', ms=7, mew=1, mec=col[0], color=col[0], linestyle='None')
  eb3[-1][0].set_linewidth(3)

  leg = plt.legend((eb1, eb3), ('Unhedged', 'Hedged'),
                   loc='upper left', borderaxespad=0.)

  plot_name = dir_figs + 'fig7.jpg'

  plt.savefig(plot_name, bbox_extra_artists=([leg]), bbox_inches='tight', dpi=1200)

  return




##########################################################################
######### calc min & avg adj rev given only contracts, no reserve, for different strikes/slopes (fig 6)####
### returns value ####
# ##########################################################################
def plot_cfd_slope_effect(dir_figs, sweWtSynth, revSimWyr, payoutCfdSim, meanRevenue, fixedCostFraction):
  netRevSimWyr = revSimWyr - meanRevenue * fixedCostFraction

  prob = 1 / sweWtSynth.shape[0]
  v_list = np.arange(0,151,2)/100

  hedgeStats = np.empty([len(v_list), 3])
  count = 0

  for v in v_list:  # swap stats as function of slope v
    adj_rev = netRevSimWyr + v * payoutCfdSim
    hedgeStats[count, :] = [v, np.mean(adj_rev), np.quantile(adj_rev, 0.05)]
    count += 1

  plt.figure()
  cmap = cm.get_cmap('viridis_r')
  cols = cmap(np.arange(2, hedgeStats.shape[0]+2) / (hedgeStats.shape[0]+2))
  cmapScalar = cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=0, vmax=1.5))
  cmapScalar._A = []
  plt.scatter(hedgeStats[:, 1], hedgeStats[:, 2], c=cols)
  plt.xlabel('Expected Hedged Net Revenue ($M/year)')
  plt.ylabel('Q05 Hedged Net Revenue ($M/year)')
  cbar = plt.colorbar(cmapScalar)
  cbar.ax.set_ylabel('Contract slope ($\$$M/inch)')

  plot_name = dir_figs + 'fig6.jpg'
  plt.savefig(plot_name, dpi=1200)

  return



##########################################################################
######### save synthetic data needed for moea ###########
############## Saves csv, no return #########################################
##########################################################################
def save_synthetic_data_moea(dir_generated_inputs, sweWtSynth, revSimWyr, payoutCfdSim):
  synthetic_data = pd.DataFrame({'swe': sweWtSynth.values, 'revenue': revSimWyr.values,
                       'payoutCfd': payoutCfdSim.values}).iloc[1:, :].reset_index(drop=True)[['swe', 'revenue', 'payoutCfd']]
  synthetic_data.to_csv(dir_generated_inputs + 'synthetic_data.txt',sep=' ', index=False)




