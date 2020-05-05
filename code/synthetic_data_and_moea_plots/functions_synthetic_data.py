##############################################################################################################
### functions_synthetic_data.py - python functions used in creating synthetic SWE, hydropower generation,
###     and power price, plus related plots
### Project started May 2017, last update Jan 2020
##############################################################################################################

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import statsmodels.formula.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sbn
import scipy as sp
from scipy import stats as st
from scipy.stats import gamma, lognorm, multivariate_normal, norm, t
from datetime import datetime
import sys
import itertools

sbn.set_style('white')
sbn.set_context('paper', font_scale=1.55)

cmap = cm.get_cmap('viridis')
col = [cmap(0.1),cmap(0.3),cmap(0.6),cmap(0.8)]

N_SAMPLES = 1000000
eps = 1e-13

##########################################################################
######### synthetic Feb & Apr SWE, with correlation preserved via copula ###########
############## Returns dataframe of Feb & Apr SWE (inch) #########################################
##########################################################################
def synthetic_swe(dir_generated_inputs, swe, redo = False, save = False):
  np.random.seed(1)
  shp_g_danFeb, dum, scl_g_danFeb = gamma.fit(swe.danFeb, floc=0)
  shp_g_danApr, dum, scl_g_danApr = gamma.fit(swe.danApr, floc=0)
  if (redo):
    ### sample from gammas using copulas
    kendallsTau = st.kendalltau(swe.danFeb, swe.danApr).correlation
    corr_norm_equiv = math.sin(kendallsTau * math.pi / 2)

    samp_fitted = multivariate_normal.rvs(mean=np.array([0, 0]), size=N_SAMPLES,
                                          cov=[[1, corr_norm_equiv],
                                               [corr_norm_equiv, 1]])
    u = norm.cdf(samp_fitted)

    sweSynth = pd.DataFrame({'danFeb': gamma.ppf(u[:, 0], a=shp_g_danFeb, loc=0, scale=scl_g_danFeb), \
                             'danApr': gamma.ppf(u[:, 1], a=shp_g_danApr, loc=0, scale=scl_g_danApr)})
    if (save):
      sweSynth.to_pickle(dir_generated_inputs + 'sweSynth.pkl')

  else:
    sweSynth = pd.read_pickle(dir_generated_inputs + 'sweSynth.pkl')

  ### check stats
  # # Kolmogorov-Smirnov test goodness of fit (if p<0.05, reject fit)
  # print(sp.stats.kstest(swe.danFeb, 'gamma', args=(shp_g_danFeb, 0, scl_g_danFeb)))
  # print(sp.stats.kstest(swe.danApr, 'gamma', args=(shp_g_danApr, 0, scl_g_danApr)))
  # # ljung-box and box-pierce tests for autocorr in original swe data (if any of p in 2nd array (L-B) or 4th array (B-P) < 0.05, reject no-autocorr)
  # print(acorr_ljungbox(swe.danFeb, lags=15, boxpierce=True))
  # print(acorr_ljungbox(swe.danApr, lags=15, boxpierce=True))
  # # test for trend in time in original swe data
  # lmSWEwTIME = sm.ols(formula='swe ~ time', data=pd.DataFrame({'swe': swe.danFeb, 'time': np.arange(1953,2017)}))
  # lmSWEwTIME = lmSWEwTIME.fit()
  # print(lmSWEwTIME.summary())
  # lmSWEwTIME = sm.ols(formula='swe ~ time', data=pd.DataFrame({'swe': swe.danApr, 'time': np.arange(1953,2017)}))
  # lmSWEwTIME = lmSWEwTIME.fit()
  # print(lmSWEwTIME.summary())

  return sweSynth



##########################################################################
######### plot of empirical vs synthetic copula for swe ###########
############## Returns figure #########################################
##########################################################################
def empirical_copula_point(point, empirical_ranks):
  p = len(point)
  n = len(empirical_ranks[0])
  j = 0
  w_i = empirical_ranks[0] >-9999999
  while (j < p):
    w_i = w_i & (empirical_ranks[j] <= point[j])
    j += 1
  w_i = np.sum(w_i) / n
  return(w_i)

def empirical_copula_many(many, empirical_ranks):
  w = many[0] * 0.
  for i in range(len(w)):
    w[i] = empirical_copula_point([many[0][i], many[1][i]], empirical_ranks)
  return(w)

def plot_empirical_synthetic_copula_swe(dir_figs, swe, startTime):
  np.random.seed(1)
  shp_g_danFeb, dum, scl_g_danFeb = gamma.fit(swe.danFeb, floc=0)
  shp_g_danApr, dum, scl_g_danApr = gamma.fit(swe.danApr, floc=0)
  # sample from gammas using copulas
  kendallsTau = st.kendalltau(swe.danFeb, swe.danApr).correlation
  corr_norm_equiv = math.sin(kendallsTau * math.pi / 2)

  samp_fitted = multivariate_normal.rvs(mean=np.array([0, 0]), size=N_SAMPLES,
                                        cov=[[1, corr_norm_equiv],
                                             [corr_norm_equiv, 1]])
  u = norm.cdf(samp_fitted)

  sweSynth = pd.DataFrame({'danFeb': gamma.ppf(u[:, 0], a=shp_g_danFeb, loc=0, scale=scl_g_danFeb), \
                           'danApr': gamma.ppf(u[:, 1], a=shp_g_danApr, loc=0, scale=scl_g_danApr)})
  # transform swe to empircal ranks
  RFeb = swe.danFeb * 0.
  RApr = swe.danFeb * 0.
  for i in range(len(RFeb)):
    RFeb.iloc[i] = (swe.danFeb <= swe.danFeb.iloc[i]).sum() / (len(RFeb) + 1)
    RApr.iloc[i] = (swe.danApr <= swe.danApr.iloc[i]).sum() / (len(RFeb) + 1)
  # empirical copula w
  w = empirical_copula_many([RFeb.values, RApr.values], [RFeb.values, RApr.values])
  nw = len(w)
  h = (nw * w - 1) / (nw - 1)
  # verify relation eqn 6, Genest&Favre 2007
  wbar = w.sum() / nw
  tau_n = 4 * nw / (nw - 1) * wbar - (nw + 3) / (nw - 1)

  ### now plot like 9b in Genest&Favre 2007, w_i vs E[w_i]
  ncop = 10000
  samp_fitted = multivariate_normal.rvs(mean=np.array([0, 0]), size=nw * ncop,
                                        cov=[[1, corr_norm_equiv],
                                             [corr_norm_equiv, 1]])
  u = norm.cdf(samp_fitted)
  samp_fitted = [pd.Series(u[:, 0]).values, pd.Series(u[:, 1]).values]

  samp_uncorr = multivariate_normal.rvs(mean=np.array([0, 0]), size=nw * ncop,
                                        cov=[[1, 0.0],
                                             [0.0, 1]])
  u = norm.cdf(samp_uncorr)
  samp_uncorr = [pd.Series(u[:, 0]).values, pd.Series(u[:, 1]).values]

  samp_corr = multivariate_normal.rvs(mean=np.array([0, 0]), size=nw * ncop,
                                      cov=[[1, 1],
                                           [1, 1]])
  u = norm.cdf(samp_corr)
  samp_corr = [pd.Series(u[:, 0]).values, pd.Series(u[:, 1]).values]

  copula_data_fitted = np.sort(empirical_copula_many([RFeb.values, RApr.values], samp_fitted))
  copula_fitted_fitted = np.zeros([ncop, nw])
  copula_uncorr_uncorr = np.zeros([ncop, nw])
  copula_corr_corr = np.zeros([ncop, nw])

  for i in range(ncop):
    copula_fitted_fitted[i, :] = np.sort(empirical_copula_many([samp_fitted[0][(i * nw):((i + 1) * nw)],
                                                         samp_fitted[1][(i * nw):((i + 1) * nw)]],
                                                        samp_fitted))
    copula_uncorr_uncorr[i, :] = np.sort(empirical_copula_many([samp_uncorr[0][(i * nw):((i + 1) * nw)],
                                                         samp_uncorr[1][(i * nw):((i + 1) * nw)]],
                                                        samp_uncorr))
    copula_corr_corr[i, :] = np.sort(empirical_copula_many([samp_corr[0][(i * nw):((i + 1) * nw)],
                                                     samp_corr[1][(i * nw):((i + 1) * nw)]],
                                                    samp_corr))
    if (i % 100 == 0):
      print('Finished copula comparison ', i+1, ' out of 10,000, ', datetime.now() - startTime)
      sys.stdout.flush()
  copula_uncorr_uncorr_avg = np.mean(copula_uncorr_uncorr, axis=0)
  copula_corr_corr_avg = np.mean(copula_corr_corr, axis=0)
  copula_fitted_fitted_avg = np.mean(copula_fitted_fitted, axis=0)
  copula_fitted_fitted_q5 = np.quantile(copula_fitted_fitted, 0.05, axis=0)
  copula_fitted_fitted_q95 = np.quantile(copula_fitted_fitted, 0.95, axis=0)

  # compare marginal dependence structure
  # plt.figure()
  # ax = plt.subplot2grid((1, 2), (0, 0))
  # ax.set_xlabel('Feb 1 SWE order statistic')
  # ax.set_ylabel('Apr 1 SWE order statistic')
  # p1 = plt.scatter(gamma.cdf(sweSynth.danFeb.iloc[:500], a=shp_g_danFeb, loc=0, scale=scl_g_danFeb),
  #                  gamma.cdf(sweSynth.danApr.iloc[:500], a=shp_g_danApr, loc=0, scale=scl_g_danApr), marker='o',
  #                  facecolors='none', linewidth=1, alpha=0.7, edgecolors=col[3], s=30)
  # p2 = plt.scatter(gamma.cdf(swe.danFeb, a=shp_g_danFeb, loc=0, scale=scl_g_danFeb),
  #                  gamma.cdf(swe.danApr, a=shp_g_danApr, loc=0, scale=scl_g_danApr), color=col[0], alpha=0.6,
  #                  marker='^', s=40)
  # ax.legend([p1, p2], ['Synthetic', 'Historic'], loc='lower right')

  plt.figure()
  plt.fill_between(copula_fitted_fitted_avg, copula_fitted_fitted_q5, copula_fitted_fitted_q95, color=col[3],
                   alpha=0.4, zorder=0)
  l1, = plt.plot(copula_fitted_fitted_avg, copula_fitted_fitted_avg, c=col[3], lw=3, zorder=1)
  p1 = plt.scatter(copula_fitted_fitted_avg, copula_data_fitted, c=col[0], zorder=2)
  plt.legend((l1, p1), ('Fitted copula', 'Observed data'))
  plt.xlabel('Theoretical order statistic')
  plt.ylabel('Observed order statistic')
  plot_name = dir_figs + 'figS1.jpg'
  plt.savefig(plot_name, dpi=1200)





##########################################################################
######### synthetic generation, based on regressions with sweFeb and sweApr ###########
############## Returns dataframe monthly gen (GWh/mnth) #########################################
##########################################################################

def synthetic_generation(dir_generated_inputs, dir_figs, gen, sweSynth, redo = False, save = False):
  np.random.seed(2)
  if (redo):
    # dum = 6
    # plt.scatter(gen.sweApr.loc[gen.wmnth == dum], gen.tot.loc[gen.wmnth == dum])

    # try linear peicewise fit, with sloped segment then flat segment
    def linear_w_max(x, intercept, slope, upperbound):
      return (np.minimum(intercept + slope * x, upperbound * np.ones(len(x))))

    # p0 = [60, 3.8, 200]
    # popt, pcov = sp.optimize.curve_fit(linear_w_max, gen.sweApr.loc[gen.wmnth == dum].values,
    #                                    gen.tot.loc[gen.wmnth == dum].values, p0)
    #
    # plt.plot(np.arange(90), linear_w_max(np.arange(90), popt[0], popt[1], popt[2]))
    # plt.scatter(gen.sweApr.loc[gen.wmnth == dum], linear_w_max(gen.sweApr.loc[gen.wmnth == dum], popt[0], popt[1], popt[2]) - gen.tot.loc[gen.wmnth == dum])



    # Store regression params and calculate predicted generation in each month
    lmGenWmnthParams = pd.DataFrame({'wmnth': [], 'int': [], 'sweFebSlp': [], 'sweAprSlp': [],
                                     'thres':[], 'residStd': []})
    gen['genPredS'] = np.nan


    # # months with significant february threshold
    # for i in [5]:
    #   # fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2)
    #   p0 = [92, 3.8, 226]
    #   popt, pcov = sp.optimize.curve_fit(linear_w_max, gen.sweFeb.loc[gen.wmnth == i].values,
    #                                      gen.tot.loc[gen.wmnth == i].values, p0)
    #   gen.genPredS.loc[gen.wmnth == i] = linear_w_max(gen.sweFeb.loc[gen.wmnth == i], popt[0], popt[1],
    #                                                   popt[2])
    #   # ax2.scatter(gen.sweFeb.loc[gen.wmnth == i], gen.tot.loc[gen.wmnth == i])
    #   # ax2.scatter(gen.sweFeb.loc[gen.wmnth == i], gen.genPredS.loc[gen.wmnth == i])
    #   # plt.scatter(gen.sweFeb.loc[gen.wmnth == i],
    #   #             gen.tot.loc[gen.wmnth == i] - gen.genPredS.loc[gen.wmnth == i])
    #   # plt.plot([(popt[2]-popt[0])/popt[1],(popt[2]-popt[0])/popt[1]],[-100,100])
    #   lmGenWmnthParams = lmGenWmnthParams.append(pd.DataFrame({'wmnth': [i], 'int': [popt[0]],
    #                                                            'sweFebSlp': [popt[1]], 'sweAprSlp': [0],
    #                                                            'thres': [popt[2]],
    #                                                            'residStd': [(gen.tot.loc[gen.wmnth == i] -
    #                                                                          gen.genPredS.loc[
    #                                                                            gen.wmnth == i]).std()]
    #                                                            })).reset_index(drop=True)

    # months with significant april threshold
    for i in [6,7,8,9]:
      # fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2)
      p0 = [92, 3.8, 226]
      popt, pcov = sp.optimize.curve_fit(linear_w_max, gen.sweApr.loc[gen.wmnth == i].values,
                                         gen.tot.loc[gen.wmnth == i].values, p0)
      gen.genPredS.loc[gen.wmnth == i] = linear_w_max(gen.sweApr.loc[gen.wmnth == i], popt[0], popt[1],
                                                      popt[2])
      # ax2.scatter(gen.sweApr.loc[gen.wmnth == i], gen.tot.loc[gen.wmnth == i])
      # ax2.scatter(gen.sweApr.loc[gen.wmnth == i], gen.genPredS.loc[gen.wmnth == i])
      # plt.scatter(gen.sweApr.loc[gen.wmnth == i],
      #             gen.tot.loc[gen.wmnth == i] - gen.genPredS.loc[gen.wmnth == i])
      # plt.plot([(popt[2]-popt[0])/popt[1],(popt[2]-popt[0])/popt[1]],[-100,100])
      lmGenWmnthParams = lmGenWmnthParams.append(pd.DataFrame({'wmnth': [i], 'int': [popt[0]],
                                                               'sweAprSlp': [popt[1]], 'sweFebSlp': [0],
                                                               'thres': [popt[2]],
                                                               'residStd': [(gen.tot.loc[gen.wmnth == i] -
                                                                             gen.genPredS.loc[
                                                                               gen.wmnth == i]).std()]
                                                               })).reset_index(drop=True)

    # months with no threshold & feb only
    for i in [2,3,4]:
      lmGenWmnth = sm.ols(formula='gen ~ swe',
                          data=pd.DataFrame(
                            {'gen': gen.tot.loc[gen.wmnth == i],
                             'swe': gen.sweFeb.loc[gen.wmnth == i]}))
      lmGenWmnth = lmGenWmnth.fit()
      # print(lmGenWmnth.summary())
      gen.genPredS.loc[gen.wmnth == i] = lmGenWmnth.params[0] + lmGenWmnth.params[1] * gen.sweFeb.loc[
        gen.wmnth == i]
      # plt.scatter(gen.sweFeb.loc[gen.wmnth == i], gen.tot.loc[gen.wmnth == i])
      # plt.scatter(gen.sweFeb.loc[gen.wmnth == i], gen.genPredS.loc[gen.wmnth == i])
      # plt.scatter(gen.sweFeb.loc[gen.wmnth == i], gen.tot.loc[gen.wmnth == i]-gen.genPredS.loc[gen.wmnth == i])
      lmGenWmnthParams = lmGenWmnthParams.append(
        pd.DataFrame({'wmnth': [i], 'int': [lmGenWmnth.params[0]],
                      'sweFebSlp': [lmGenWmnth.params[1]],
                      'sweAprSlp': [0],
                      'thres': [1000],
                      'residStd': [lmGenWmnth.resid.std()]})).reset_index(drop=True)

    # months with no threshold & apr
    for i in [5,10,11]:
      lmGenWmnth = sm.ols(formula='gen ~ swe',
                          data=pd.DataFrame(
                            {'gen': gen.tot.loc[gen.wmnth == i],
                             'swe': gen.sweApr.loc[gen.wmnth == i]}))
      lmGenWmnth = lmGenWmnth.fit()
      # print(lmGenWmnth.summary())
      gen.genPredS.loc[gen.wmnth == i] = lmGenWmnth.params[0] + lmGenWmnth.params[1] * gen.sweApr.loc[
        gen.wmnth == i]
      # plt.scatter(gen.sweApr.loc[gen.wmnth == i], gen.tot.loc[gen.wmnth == i])
      # plt.scatter(gen.sweApr.loc[gen.wmnth == i], gen.genPredS.loc[gen.wmnth == i])
      # plt.scatter(gen.sweApr.loc[gen.wmnth == i], gen.tot.loc[gen.wmnth == i] - gen.genPredS.loc[gen.wmnth == i])
      lmGenWmnthParams = lmGenWmnthParams.append(
        pd.DataFrame({'wmnth': [i], 'int': [lmGenWmnth.params[0]],
                      'sweFebSlp': [0],
                      'sweAprSlp': [lmGenWmnth.params[1]],
                      'thres': [1000],
                      'residStd': [lmGenWmnth.resid.std()]})).reset_index(drop=True)

    # months with no threshold or swe
    for i in [1,12]:
      gen.genPredS.loc[gen.wmnth == i] = gen.tot.loc[gen.wmnth == i].mean()
      lmGenWmnthParams = lmGenWmnthParams.append(
        pd.DataFrame({'wmnth': [i], 'int': [gen.tot.loc[gen.wmnth == i].mean()],
                      'sweFebSlp': [0],
                      'sweAprSlp': [0],
                      'thres': [1000],
                      'residStd': [(gen.tot.loc[gen.wmnth == i] -
                                    gen.tot.loc[gen.wmnth == i].mean()).std()]})).reset_index(drop=True)



    ### plot 12 monthly models with data (Fig S2)
    max_x = 60
    wmnths = ['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']
    plt.figure()
    for i in range(3):
      for j in range(4):
        ax = plt.subplot2grid((3, 4), (i,j))
        if (j != 0):
          ax.tick_params(axis='y', which='both', labelleft=False)
        if (i != 2):
          ax.tick_params(axis='x', which='both', labelbottom=False)
        if (i == 2) & (j == 1):
          ax.set_xlabel('                        Predictor SWE (inches)')
        elif (i == 1) & (j == 0):
          ax.set_ylabel('Generation (GWh/month)')
        # ax.xaxis.set_label_position('top')
        # ax.set_xticks(np.arange(-2, 6, 7))
        ax.set_xlim([0,max_x])
        ax.set_ylim([0,280])
        ax.set_xticks(np.arange(0, 51, 25))
        ax.set_yticks(np.arange(0, 201, 100))
        wmnth = 1 + 4*i + j
        if (wmnth <= 4):
          swetemp = gen.sweFeb.loc[gen.wmnth == wmnth]
        else:
          swetemp = gen.sweApr.loc[gen.wmnth == wmnth]
        plt.scatter(swetemp, gen.tot.loc[gen.wmnth == wmnth], c=col[3])
        x0 = 0
        y0 = lmGenWmnthParams.int.loc[lmGenWmnthParams.wmnth==wmnth].iloc[0]
        y1 = lmGenWmnthParams.thres.loc[lmGenWmnthParams.wmnth==wmnth].iloc[0]
        slp = (lmGenWmnthParams.sweAprSlp.loc[lmGenWmnthParams.wmnth==wmnth].iloc[0] +
                               lmGenWmnthParams.sweFebSlp.loc[lmGenWmnthParams.wmnth==wmnth].iloc[0])
        x1 = (y1 - y0) / slp
        if (slp == 0):
          plt.axhline(y0, c=col[0])
        else:
          plt.plot([x0, x1], [y0, y1], c=col[0])
        if (x1 < max_x):
          plt.plot([x1, max_x], [y1, y1], c=col[0])
        plt.annotate(wmnths[wmnth-1], xy=(38,3))
    plot_name = dir_figs + 'figS2.jpg'
    plt.savefig(plot_name, dpi=1200)


    gen['genResidS'] = gen.tot - gen.genPredS

    # # plot hist and prediction
    # plt.plot(gen.tot)
    # plt.plot(gen.genPredS)
    # plt.plot(gen.genResidS)
    # pd.plotting.autocorrelation_plot(gen.genResidS)
    # plt.hist(gen.genResidS)
    # plt.scatter(gen.sweFeb,gen.genResidS)
    # plt.scatter(gen.sweApr,gen.genResidS)
    # plt.scatter(gen.wmnth,gen.genResidS)

    # check autocorrelation -> highly autocorr
    # print(stm.stats.acorr_ljungbox(gen.genResidS, lags=60, boxpierce=True))

    ### now deseasonalize, also accounting for lower residuals above threshold
    gen['genResidSDe'] = np.nan
    for i in range(1, 13):
      if (lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] > 999):
        gen.genResidSDe.loc[gen.wmnth == i] = (gen.genResidS.loc[gen.wmnth == i] - gen.genResidS.loc[gen.wmnth == i].mean()) / gen.genResidS.loc[gen.wmnth == i].std()
      else:
        gen.genResidSDe.loc[(gen.wmnth == i) & (
                gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)] \
          = (gen.genResidS.loc[(gen.wmnth == i) & (
                gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)]
             - gen.genResidS.loc[(gen.wmnth == i) & (
                        gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)].mean()) \
            / gen.genResidS.loc[(gen.wmnth == i) & (
                gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)].std()

        gen.genResidSDe.loc[(gen.wmnth == i) & (
                gen.genPredS < lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)] \
          = (gen.genResidS.loc[(gen.wmnth == i) & (
                gen.genPredS < lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)]
             - gen.genResidS.loc[(gen.wmnth == i) & (
                        gen.genPredS < lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)].mean()) \
            / gen.genResidS.loc[(gen.wmnth == i) & (
                gen.genPredS < lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)].std()

    # plt.plot(gen.genResidSDe)
    # plt.scatter(gen.wmnth, gen.genResidSDe)
    # plt.scatter(gen.sweApr, gen.genResidSDe)
    # plt.scatter(gen.sweApr, gen.genResidS)
    #
    # sp.stats.shapiro(gen.genResidSDe)
    # stt.durbin_watson(gen.genResidSDe)
    # plt.hist(gen.genResidSDe)
    # pd.plotting.autocorrelation_plot(gen.genResidSDe)
    # print(stm.stats.acorr_ljungbox(gen.genResidSDe, lags=60, boxpierce=True))


    ## now fit AR model to deseasonalized resids
    # lmGenAR = sm.ols(formula='dat ~ dat_1 +  dat_3+ dat_6-1', data = pd.DataFrame({'dat': gen.genResidSDe.iloc[12:].reset_index(drop=True),
    #                                                                          'dat_1': gen.genResidSDe.iloc[11:-1].reset_index(drop=True),
    #                                                                          'dat_2': gen.genResidSDe.iloc[10:-2].reset_index(drop=True),
    #                                                                          'dat_3': gen.genResidSDe.iloc[9:-3].reset_index(drop=True),
    #                                                                          'dat_4': gen.genResidSDe.iloc[8:-4].reset_index(drop=True),
    #                                                                          'dat_6': gen.genResidSDe.iloc[6:-6].reset_index(drop=True),
    #                                                                          'dat_12': gen.genResidSDe.iloc[:-12].reset_index(drop=True)}))
    # lmGenAR = sm.ols(formula='dat ~ dat_1 +dat_3 + dat_4 -1', data = pd.DataFrame({'dat': gen.genResidSDe.iloc[4:].reset_index(drop=True),
    #                                                                          'dat_1': gen.genResidSDe.iloc[3:-1].reset_index(drop=True),
    #                                                                          'dat_2': gen.genResidSDe.iloc[2:-2].reset_index(drop=True),
    #                                                                          'dat_3': gen.genResidSDe.iloc[1:-3].reset_index(drop=True),
    #                                                                          'dat_4': gen.genResidSDe.iloc[:-4].reset_index(drop=True)}))
    lmGenAR = sm.ols(formula='dat ~ dat_1 +dat_3 -1',
                     data=pd.DataFrame({'dat': gen.genResidSDe.iloc[3:].reset_index(drop=True),
                                        'dat_1': gen.genResidSDe.iloc[2:-1].reset_index(drop=True),
                                        'dat_2': gen.genResidSDe.iloc[1:-2].reset_index(drop=True),
                                        'dat_3': gen.genResidSDe.iloc[:-3].reset_index(drop=True)}))
    lmGenAR = lmGenAR.fit()
    # print(lmGenAR.summary())

    ## resids from AR(1,3) model
    gen['genResidSDeAR'] = np.nan
    for i in range(3, gen.shape[0]):
      gen.genResidSDeAR.iloc[i] = gen.genResidSDe.iloc[i] - lmGenAR.params[0] * gen.genResidSDe.iloc[i - 1] - \
                                  lmGenAR.params[1] * gen.genResidSDe.iloc[i - 3]

    # sp.stats.shapiro(gen.genResidSDeAR.iloc[3:])
    # stt.durbin_watson(gen.genResidSDeAR.iloc[3:])
    # stm.stats.acorr_ljungbox(gen.genResidSDeAR.iloc[3:], boxpierce=True, lags=36)
    # plt.hist(gen.genResidSDeAR.iloc[3:])
    # pd.plotting.autocorrelation_plot(gen.genResidSDeAR.iloc[4:])
    # st.probplot(gen.genResidSDeAR.iloc[3:].loc[gen.wmnth == 12], plot=plt)
    # plt.scatter( gen.wmnth.iloc[4:],gen.genResidSDeAR.iloc[4:])

    # # test for normality of each month's residuals
    # i = 12
    # print(st.normaltest(gen.genResidSDeAR.iloc[3:].loc[gen.wmnth == i]))




    ### Simulate new hydro gen
    AR_mean = 0  # lmGenAR.resid.mean()
    AR_std = lmGenAR.resid.std()
    residAR1_wt = lmGenAR.params[0]
    residAR3_wt = lmGenAR.params[1]

    # do iterative parts in numpy for speed
    dum = np.full(((N_SAMPLES + 1) * 12, 6), -100.0)
    dum[:, 0] = norm.rvs(AR_mean, AR_std, (N_SAMPLES + 1) * 12)  # col 0 = residSDeAR (normal residuals from AR process)
    dum[:3, 1] = norm.rvs(AR_mean, AR_std, 3)  # col 1 = residSDe (deseas resids from snow reg, after applying AR)(start with random b4 burn in)
    for i in range(3, dum.shape[0]):
      dum[i, 1] = residAR1_wt * dum[i - 1, 1] + residAR3_wt * dum[i - 3, 1] + dum[i, 0]
    dum = dum[12:, :]   # get rid of burn-in
    snowFeb = sweSynth.danFeb
    snowApr = sweSynth.danApr  # from correlated gammas, see below

    for i in range(0, N_SAMPLES):
      dum[(12 * i):(12 * (i + 1)), 2] = i  # col 2 = wyr
      dum[(12 * i):(12 * (i + 1)), 3] = range(1, 13)  # col 3 = wmnth
      dum[(12 * i):(12 * (i + 1)), 4] = snowFeb[i]  # col 4 = Feb snow val
      dum[(12 * i):(12 * (i + 1)), 5] = snowApr[i]  # col 5 = Apr snow val

    # now get dataframe and calc rest of sim vars
    genSynth = pd.DataFrame(
      {'wyr': dum[:, 2], 'wmnth': dum[:, 3], 'sweFeb': dum[:, 4], 'sweApr': dum[:, 5], 'residSDe': dum[:, 1],
       'residS': np.nan, 'genPred': np.nan, 'gen': np.nan})
    genSynth.wmnth = genSynth.wmnth.apply(int)
    genSynth.wyr = genSynth.wyr.apply(int)

    # get prediction from monthly gen~snow regressions, and synthetic gen by adding residS.
    for i in range(1, 13):
      genSynth.genPred.loc[genSynth.wmnth == i] = np.minimum((lmGenWmnthParams.int.loc[lmGenWmnthParams.wmnth == i].iloc[0] +
                                                              lmGenWmnthParams.sweFebSlp.loc[
                                                                lmGenWmnthParams.wmnth == i].iloc[
                                                                0] * genSynth.sweFeb.loc[genSynth.wmnth == i] +
                                                              lmGenWmnthParams.sweAprSlp.loc[
                                                                lmGenWmnthParams.wmnth == i].iloc[
                                                                0] * genSynth.sweApr.loc[genSynth.wmnth == i]).values,
                                                             lmGenWmnthParams.thres.loc[lmGenWmnthParams.wmnth == i].iloc[0])

    # now reseasonalize autocorrelated residual variance. result is residual from monthly gen~snow regressions,
    #  accounting for lower residuals above thresholds
    if (lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] > 999):
      gen.genResidSDe.loc[gen.wmnth == i] = (gen.genResidS.loc[gen.wmnth == i] - gen.genResidS.loc[
        gen.wmnth == i].mean()) / gen.genResidS.loc[gen.wmnth == i].std()
    else:
      gen.genResidSDe.loc[(gen.wmnth == i) & (
              gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)] \
        = (gen.genResidS.loc[(gen.wmnth == i) & (
              gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)]
           - gen.genResidS.loc[(gen.wmnth == i) & (
                      gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)].mean()) \
          / gen.genResidS.loc[(gen.wmnth == i) & (
              gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)].std()

    for i in range(1, 13):
      if (lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] > 999):
        genSynth.residS.loc[genSynth.wmnth == i] = genSynth.residSDe.loc[genSynth.wmnth == i] * \
                                                   gen.genResidS.loc[gen.wmnth == i].std() + \
                                                   gen.genResidS.loc[gen.wmnth == i].mean()
      else:
        genSynth.residS.loc[(genSynth.wmnth == i) & (
                genSynth.genPred > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)] \
          = genSynth.residSDe.loc[(genSynth.wmnth == i) & (
                genSynth.genPred > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)] * \
            gen.genResidS.loc[(gen.wmnth == i) & (
                    gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)].std() + \
            gen.genResidS.loc[(gen.wmnth == i) & (
                    gen.genPredS > lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[
              0] - eps)].mean()
        genSynth.residS.loc[(genSynth.wmnth == i) & (
                genSynth.genPred < lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)] \
          = genSynth.residSDe.loc[(genSynth.wmnth == i) & (
                genSynth.genPred < lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)] * \
            gen.genResidS.loc[(gen.wmnth == i) & (
                    gen.genPredS < lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[0] - eps)].std() + \
            gen.genResidS.loc[(gen.wmnth == i) & (
                    gen.genPredS < lmGenWmnthParams.loc[lmGenWmnthParams.wmnth == i].thres.values[
              0] - eps)].mean()

    genSynth['gen'] = genSynth.genPred + genSynth.residS

    # make sure synthetic between historical limits, reflecting minimum releases & max turbine capacity
    genSynth['gen'].loc[genSynth.gen < gen.tot.min()] = gen.tot.min()
    genSynth['gen'].loc[genSynth.gen > gen.tot.max()] = gen.tot.max()

    genSynth = genSynth.loc[:,['wyr','wmnth','sweFeb','sweApr','gen','genPred']]

    if (save):
      genSynth.to_pickle(dir_generated_inputs + 'genSynth.pkl')


  else:
    genSynth = pd.read_pickle(dir_generated_inputs + 'genSynth.pkl')


  ### check stats, compare synthetic to historical
  # # compare monthly trends to observed record
  # genMonths = pd.DataFrame({'wmnth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  #                           'mean': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
  #                             lambda x: gen['tot'].loc[gen['wmnth'] == x].mean()),
  #                           'std': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
  #                             lambda x: gen['tot'].loc[gen['wmnth'] == x].std())})
  # genMonths['cv'] = genMonths['std'] / genMonths['mean']
  # genMonths['simMean'] = pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
  #   lambda x: genSynth['gen'].loc[genSynth['wmnth'] == x].mean())
  # genMonths['simStd'] = pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
  #   lambda x: genSynth['gen'].loc[genSynth['wmnth'] == x].std())
  #
  # plt.plot(genMonths['mean'])
  # plt.plot(genMonths['simMean'])
  # plt.plot(genMonths['std'])
  # plt.plot(genMonths['simStd'])
  # # # make sure annual correlations still hold
  # genSynthWyr = genSynth.groupby('wyr').sum().gen
  # genWyr = gen.groupby('wyear').sum().tot
  # np.mean(genWyr)
  # np.mean(genSynthWyr)
  # (np.mean(genSynthWyr) - np.mean(genWyr)) / np.mean(genWyr)
  # np.std(genWyr)
  # np.std(genSynthWyr)
  # (np.std(genSynthWyr) - np.std(genWyr)) / np.std(genWyr)
  #
  # # check whether deviation from seasonal expectation (based on swe) reflects historical distribution
  # if (redo):
  #   print(st.ks_2samp(genSynth.gen.iloc[:360] - genSynth.genPred.iloc[:360], gen.tot - gen.genPredS))
  #   print(st.ks_2samp(genSynth.gen - genSynth.genPred, gen.tot - gen.genPredS))
  #
  # # check whether gen itself comes from same distribution
  # print(st.ks_2samp(genSynth.gen, gen.tot))
  # # check whether yearly gen is from same distribution
  # print(st.ks_2samp(genSynthWyr, genWyr))

  return (genSynth)







##########################################################################
######### synthetic power price, based on synth gas price ###########
############## Returns dataframe monthly power price ($/MWh) #########################################
##########################################################################

def synthetic_power(dir_generated_inputs, power, redo = False, save = False):
  np.random.seed(3)
  if (redo):

    # log-transform and deseasonalize
    power['logMean'] = np.log(power.priceMean)
    power['logDe'] = np.nan
    for i in range(1, 13):
      power.logDe.loc[power.wmnth == i] = (power.logMean.loc[power.wmnth == i] -
                                           power.logMean.loc[power.wmnth == i].mean()) / \
                                          power.logMean.loc[power.wmnth == i].std()

    # plt.plot(power.logMean)
    # plt.plot(power.logDe)

    # # # check for linear trend -> small significant negative trend. ignore since only 7 years of data.
    # lmPowDeLin = sm.ols(formula='dat ~ ind ',
    #                   data=pd.DataFrame({'dat': power.logDe, 'ind': range(0, power.shape[0])}))
    # lmPowDeLin = lmPowDeLin.fit()
    # print(lmPowDeLin.summary())

    # ### SARIMAX model: iterate over parameters and choose lowest BIC
    # # # (mod from https://stats.stackexchange.com/questions/328524/choose-seasonal-parameters-for-sarimax-model)
    # p = d = q = P = D = Q = range(0,2)
    # pdq = list(itertools.product(p,d,q))
    # PDQ12 = [(x[0], x[1], x[2], 12) for x in list(itertools.product(P,D,Q))]
    # BIC = 1000
    # for param in pdq:
    #     for paramSeas in PDQ12:
    #         try:
    #             sarimaxPower = SARIMAX(power.logDe, order=param, seasonal_order=paramSeas)
    #             sarimaxPower = sarimaxPower.fit(disp=0)
    #             if sarimaxPower.bic < 124:
    #                 print('ARIMA{}x{} - BIC:{}'.format(param, paramSeas, sarimaxPower.bic))
    #             if sarimaxPower.bic < BIC:
    #                 BIC = sarimaxPower.bic
    #                 best_param = param
    #                 best_paramSeas = paramSeas
    #         except Exception as e:
    #             # print(e)
    #             continue
    # sarimaxPower = SARIMAX(power.logDe, order=(1,0,0), seasonal_order=(0,0,1,12))
    # sarimaxPower = sarimaxPower.fit(disp=0)
    # # print(sarimaxPower.summary())

    # p = q = P = Q = range(0, 2)
    # pdq = [(x[0], 0, x[1]) for x in list(itertools.product(p, q))]
    # PDQ12 = [(x[0], 0, x[1], 12) for x in list(itertools.product(P, Q))]
    # BIC = 1000
    # for param in pdq:
    #   for paramSeas in PDQ12:
    #     try:
    #       sarimaxPower = SARIMAX(power.logDe, order=param, seasonal_order=paramSeas)
    #       sarimaxPower = sarimaxPower.fit(disp=0)
    #       # if ((sarimaxPower.pvalues > 0.05).sum() == 0):
    #         # if sarimaxPower.bic < 115:
    #       print('ARIMA{}x{} - BIC:{}'.format(param, paramSeas, sarimaxPower.bic))
    #       if sarimaxPower.bic < BIC:
    #         BIC = sarimaxPower.bic
    #         best_param = param
    #         best_paramSeas = paramSeas
    #     except Exception as e:
    #       # print(e)
    #       continue
    sarimaxPower = SARIMAX(power.logDe, order=(1, 0, 0), seasonal_order=(0, 0, 1, 12))
    sarimaxPower = sarimaxPower.fit(disp=0)
    print(sarimaxPower.summary())



    # # try with sweApr as exogenous factor -> not sig
    # power['wyr'] = power.index.year
    # power.wyr.loc[power.wmnth < 4] = power.wyr.loc[power.wmnth < 4] + 1
    # power['swe'] = np.nan
    # for i in range(2010, 2018):
    #     power.swe.loc[power.wyr == i] = swe.danApr[i]
    # sarimaxPower = SARIMAX(power.logDe, exog=power.swe, order=(1,0,0), seasonal_order=(0,0,1,12))
    # sarimaxPower = sarimaxPower.fit(disp=0)
    # print(sarimaxPower.summary())
    #
    # # try with snow year type as exog -> not sig
    # power['sweAprThirds'] = 1
    # power.sweAprThirds.loc[power.swe > swe.danApr.quantile(0.67)] = 2
    # power.sweAprThirds.loc[power.swe < swe.danApr.quantile(0.33)] = 0
    # sarimaxPower = SARIMAX(power.logDe, exog=power.sweAprThirds, order=(1, 0, 0), seasonal_order=(0, 0, 1, 12))
    # sarimaxPower = sarimaxPower.fit(disp=0)
    # print(sarimaxPower.summary())

    ### check stats, plots
    # plt.plot(sarimaxPower.resid.iloc[12:])
    # plt.hist(sarimaxPower.resid.iloc[12:])
    # pd.plotting.autocorrelation_plot(sarimaxPower.resid.iloc[12:])
    # plot_pacf(sarimaxPower.resid.iloc[12:])
    # acorr_ljungbox(sarimaxPower.resid.iloc[12:], boxpierce=True, lags=36)
    # sp.stats.shapiro(sarimaxPower.resid.iloc[12:])
    # stt.durbin_watson(sarimaxPower.resid.iloc[12:])
    # plt.plot(sarimaxPower.predict().iloc[12:])
    # plt.plot(power.logDe.iloc[12:])
    # plt.scatter(power.wmnth.iloc[12:], sarimaxPower.resid.iloc[12:])
    # plt.scatter(power.wmnth.iloc[12:], power.logDe.iloc[12:])


    ### Simulate new power prices
    logDeAR1coef = sarimaxPower.params[0]
    logDeMA12coef = sarimaxPower.params[1]
    logDeERRSTD = np.std(sarimaxPower.resid) # np.sqrt(sarimaxPower.params[2])


    # Calc random aspects of power sim. Serial calcs in numpy.
    burn=4
    dum = np.full(((N_SAMPLES + burn) * 12, 4), -100.0)
    dum[:12, 2] = power.logDe.iloc[-12:].values          ## start with oct2015-sep2016, and burn in 2 extra yrs (total 4).
    dum[:12, 3] = sarimaxPower.resid.iloc[-12:].values
    for i in range(0, N_SAMPLES + burn):
      dum[(12 * i):(12 * (i + 1)), 0] = i - burn  # col 0 = wyr
      dum[(12 * i):(12 * (i + 1)), 1] = [1,2,3,4,5,6,7,8,9,10,11,12]  # col 1 = wmnth
    dum[12:, 3] = norm.rvs(0, logDeERRSTD, (N_SAMPLES + burn - 1) * 12)  # col 3 = resids from SARMA model -> normal
    for i in range(12, (N_SAMPLES + burn) * 12):
      dum[i, 2] = logDeAR1coef * dum[i - 1, 2] + \
                  logDeMA12coef * dum[i - 12, 3] + \
                  dum[i, 3]  # # col 2 = deseasonalized log power price

    # plt.plot(range(84,84+4800),dum[:4800,2])
    # plt.plot(power.logDe.values)
    dum = dum[(12 * burn):, :]

    # Set in dataframe and calc rest of sim variables
    powSynth = pd.DataFrame({'wyr': dum[:, 0], 'wmnth': dum[:, 1], 'logDe': dum[:, 2]})

    powSynth['logPrice'] = np.nan
    for i in range(1, 13):
      powSynth.logPrice.loc[powSynth.wmnth == i] = powSynth.logDe.loc[powSynth.wmnth == i] * \
                                                   power.logMean.loc[power.wmnth == i].std() + \
                                                   power.logMean.loc[power.wmnth == i].mean()

    powSynth['powPrice'] = np.exp(powSynth.logPrice)

    ### check stats, plots
    # powSynth.powPrice.mean()
    # power.priceMean.mean()
    # powSynth.powPrice.std()
    # power.priceMean.std()
    # powSynth.powPrice.loc[powSynth.wmnth == 5].std()
    # power.priceMean.loc[power.wmnth == 5].std()
    # st.probplot(powSynth.powPrice, plot=plt)
    # st.probplot(power.priceMean, plot=plt)

    # plt.plot(powSynth.groupby('wmnth').mean().powLog)
    # plt.plot(power.groupby('wmnth').mean().logMean)
    # plt.plot(powSynth.groupby('wmnth').std().powLog)
    # plt.plot(power.groupby('wmnth').std().logMean)

    # # check whether power price comes from same distribution
    # print(st.ks_2samp(powSynth.powPrice, power.priceMean))
    # # also check yearly stats
    # power['wyr'] = power.index.year
    # power.wyr.loc[power.wmnth < 4] = power.wyr.loc[power.wmnth < 4] + 1
    # print(st.ks_2samp(powSynth.groupby('wyr').mean().powPrice, power.groupby('wyr').mean().priceMean))

    powSynth = powSynth.loc[:, ['wyr', 'wmnth', 'powPrice']]

    if (save):
      powSynth.to_pickle(dir_generated_inputs + 'powSynth.pkl')

  else:
    powSynth = pd.read_pickle(dir_generated_inputs + 'powSynth.pkl')


  return powSynth



##########################################################################
######### plot historical vs synthetic hydro generation and power prices (fig 4)###########
############## Returns figure #########################################
##########################################################################
def plot_historical_synthetic_generation_power(dir_figs, gen, genSynth, power, powSynth, genOnly, genCombined, powerOnly):
  # plot_name: include directory, no extension
  plt.figure()
  if (powerOnly == False):
    if (genCombined):
      # compare monthly trends to observed record
      genMonths = pd.DataFrame({'wmnth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                'meanHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                  lambda x: gen['tot'].loc[gen['wmnth'] == x].mean()),
                                'stdHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                  lambda x: gen['tot'].loc[gen['wmnth'] == x].std()),
                                'meanSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                  lambda x: genSynth['gen'].loc[genSynth['wmnth'] == x].mean()),
                                'stdSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                  lambda x: genSynth['gen'].loc[genSynth['wmnth'] == x].std())
                                })

      ax = plt.subplot2grid((1, 1), (0, 0))
      ax.set_xlabel('Month')
      ax.set_ylabel('Generation (GWh/month)')
      ax.set_xlim([-0.5, 11.5])
      ax.set_xticks(genMonths.index, ['O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S'])

      eb1 = ax.errorbar(genMonths.index - 0.075, genMonths.meanHist,
                         yerr=genMonths.stdHist, color=col[0], marker = 's', ms=3, mew=1, mec=col[0], linestyle='None')
      eb1[-1][0].set_linewidth(2)
      eb2 = ax.errorbar(genMonths.index + 0.075, genMonths.meanSynth,
                         yerr=genMonths.stdSynth, color=col[0], marker = 's', markerfacecolor='white',
                         markeredgecolor=col[0], mew=1, ms=3,linestyle='None')
      eb2[-1][0].set_linestyle('--')
      eb2[-1][0].set_linewidth(2)
      for i in range(12):
        ax.axvline(i + 0.5, color='lightgrey', lw=1)

      leg1 = plt.legend((eb1, eb2), ('Historic', 'Synthetic'), loc='upper left')

    else:
      # plot boxplot of generation for each wmnth
      gen['sweAprThirds'] = 'average'
      gen.sweAprThirds.loc[gen.sweApr > gen.sweWt.quantile(0.67)] = 'wet'
      gen.sweAprThirds.loc[gen.sweApr < gen.sweWt.quantile(0.33)] = 'dry'
      my_palette = {'wet': col[0], 'average': col[2], 'dry': col[3]}

      genSynth['sweAprThirds'] = 'average'
      genSynth.sweAprThirds.loc[genSynth.sweApr > gen.sweWt.quantile(0.67)] = 'wet'
      genSynth.sweAprThirds.loc[genSynth.sweApr < gen.sweWt.quantile(0.33)] = 'dry'

      genMonthsDry = pd.DataFrame({'wmnth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                   'meanHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                     lambda x: gen['tot'].loc[(gen['wmnth'] == x) & (gen.sweAprThirds == 'dry')].mean())/1000,
                                   'stdHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                     lambda x: gen['tot'].loc[(gen['wmnth'] == x) & (gen.sweAprThirds == 'dry')].std())/1000,
                                   'meanSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                     lambda x: genSynth['gen'].loc[(genSynth['wmnth'] == x) & (genSynth.sweAprThirds == 'dry')].mean())/1000,
                                   'stdSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                     lambda x: genSynth['gen'].loc[(genSynth['wmnth'] == x) & (genSynth.sweAprThirds == 'dry')].std())/1000,
                                   })
      genMonthsWet = pd.DataFrame({'wmnth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                   'meanHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                     lambda x: gen['tot'].loc[(gen['wmnth'] == x) & (gen.sweAprThirds == 'wet')].mean())/1000,
                                   'stdHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                     lambda x: gen['tot'].loc[(gen['wmnth'] == x) & (gen.sweAprThirds == 'wet')].std())/1000,
                                   'meanSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                     lambda x: genSynth['gen'].loc[(genSynth['wmnth'] == x) & (genSynth.sweAprThirds == 'wet')].mean())/1000,
                                   'stdSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                     lambda x: genSynth['gen'].loc[(genSynth['wmnth'] == x) & (genSynth.sweAprThirds == 'wet')].std())/1000,
                                   })
      genMonthsAverage = pd.DataFrame({'wmnth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                       'meanHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                         lambda x: gen['tot'].loc[(gen['wmnth'] == x) & (gen.sweAprThirds == 'average')].mean())/1000,
                                       'stdHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                         lambda x: gen['tot'].loc[(gen['wmnth'] == x) & (gen.sweAprThirds == 'average')].std())/1000,
                                       'meanSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                         lambda x: genSynth['gen'].loc[(genSynth['wmnth'] == x) & (genSynth.sweAprThirds == 'average')].mean())/1000,
                                       'stdSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                         lambda x: genSynth['gen'].loc[(genSynth['wmnth'] == x) & (genSynth.sweAprThirds == 'average')].std())/1000,
                                       })

      #  plot monthly ranges with wet-avg-dry separated, std as error bars
      if (genOnly == False):
        ax = plt.subplot2grid((2,1), (0, 0))
        ax.tick_params(axis='x', which='both', labelbottom=False)
      else:
        ax = plt.subplot2grid((1,1),(0,0))
      ax.set_ylabel('Generation (TWh/month)')
      ax.set_xlim([-0.5,11.5])
      ax.set_xticks(genMonthsWet.index, ['O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S'])
      eb1 = ax.errorbar(genMonthsAverage.index-0.075, genMonthsAverage.meanHist,
                         yerr=genMonthsAverage.stdHist, color=my_palette['average'], marker = '^', ms=4, mew=1, mec=my_palette['average'],
                         linestyle='None')
      eb1[-1][0].set_linewidth(2)
      eb2 = ax.errorbar(genMonthsAverage.index+0.075, genMonthsAverage.meanSynth,
                         yerr=genMonthsAverage.stdSynth, color=my_palette['average'], marker = '^', markerfacecolor='white',
                         markeredgecolor=my_palette['average'], mew=1, ms=4,linestyle='None')
      eb2[-1][0].set_linestyle('--')
      eb2[-1][0].set_linewidth(2)
      eb3 = ax.errorbar(genMonthsDry.index-0.375, genMonthsDry.meanHist,
                         yerr=genMonthsDry.stdHist, color=my_palette['dry'], marker = 's', ms=4, mew=1, mec=my_palette['dry'],
                         linestyle='None')
      eb3[-1][0].set_linewidth(2)
      eb4 = ax.errorbar(genMonthsDry.index-0.225, genMonthsDry.meanSynth,
                         yerr=genMonthsDry.stdSynth, color=my_palette['dry'], marker = 's', markerfacecolor='white',
                         markeredgecolor=my_palette['dry'], mew=1, ms=4,linestyle='None')
      eb4[-1][0].set_linestyle('--')
      eb4[-1][0].set_linewidth(2)
      eb5 = ax.errorbar(genMonthsWet.index+0.225, genMonthsWet.meanHist,
                         yerr=genMonthsWet.stdHist, color=my_palette['wet'], marker = 'o', ms=4, mew=1, mec=my_palette['wet'],
                         linestyle='None')
      eb5[-1][0].set_linewidth(2)
      eb6 = ax.errorbar(genMonthsWet.index + 0.375, genMonthsWet.meanSynth,
                         yerr=genMonthsWet.stdSynth, color=my_palette['wet'], marker = 'o', markerfacecolor='white',
                         markeredgecolor=my_palette['wet'], mew=1, ms=4,linestyle='None')
      eb6[-1][0].set_linestyle('--')
      eb6[-1][0].set_linewidth(2)
      for i in range(12):
        ax.axvline(i + 0.5, color='lightgrey', lw=1)

      if (genOnly == False):
        leg1 = ax.legend((eb3,eb4,eb1,eb2,eb5,eb6),
                  ('Dry Historic','Dry Synthetic','Avg Historic','Avg Synthetic','Wet Historic','Wet Synthetic'),
                  bbox_to_anchor=(0.99, 0.95), ncol=1, borderaxespad=0.)
      else:
        leg1 = ax.legend((eb3,eb4,eb1,eb2,eb5,eb6),
                  ('Dry Historic','Dry Synthetic','Avg Historic','Avg Synthetic','Wet Historic','Wet Synthetic'),
                  loc='upper left', borderaxespad=0.)
  if (genOnly == False):
    # now plot historical vs synthetic power prices
    powMonths = pd.DataFrame({'wmnth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                              'meanHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                lambda x: power['priceMean'].loc[power['wmnth'] == x].mean()),
                              'stdHist': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                lambda x: power['priceMean'].loc[power['wmnth'] == x].std()),
                              'meanSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                lambda x: powSynth['powPrice'].loc[powSynth['wmnth'] == x].mean()),
                              'stdSynth': pd.DataFrame({'dum': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})['dum'].apply(
                                lambda x: powSynth['powPrice'].loc[powSynth['wmnth'] == x].std()),
                              })

    if (powerOnly == False):
      ax = plt.subplot2grid((2,1), (1,0))
    else:
      ax = plt.subplot2grid((1,1),(0,0))
    ax.set_xlabel('Month')
    ax.set_ylabel('Price ($/MWh)')
    ax.set_xlim([-0.5,11.5])

    ax.set_xticks(powMonths.index)
    ax.set_xticklabels(['O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S'])

    eb1 = ax.errorbar(powMonths.index - 0.1, powMonths.meanHist,
                       yerr=powMonths.stdHist, color=col[0], marker = 's', ms=4, mew=1, mec=col[0], linestyle='None')
    eb1[-1][0].set_linewidth(2)
    eb2 = ax.errorbar(powMonths.index + 0.1, powMonths.meanSynth,
                       yerr=powMonths.stdSynth, color=col[0], marker = 's', markerfacecolor='white',
                       markeredgecolor=col[0], mew=1, ms=4,linestyle='None')
    eb2[-1][0].set_linestyle('--')
    eb2[-1][0].set_linewidth(2)

    for i in range(12):
      ax.axvline(i + 0.5, color='lightgrey', lw=1)

    if (powerOnly == False):
      leg2 = ax.legend((eb1, eb2), ('Historic', 'Synthetic'), bbox_to_anchor=(0.99, 0.68), ncol=1, borderaxespad=0.)
    else:
      leg2 = ax.legend((eb1, eb2), ('Historic', 'Synthetic'), loc='upper right', borderaxespad=0.)

  if ((powerOnly==False) & (genOnly==False)):
    plot_name = dir_figs + 'fig4.jpg'
    plt.savefig(plot_name, bbox_extra_artists=([leg1, leg2]), bbox_inches='tight', dpi=1200)
  elif (powerOnly == False):
    plot_name = dir_figs + 'historical_synthetic_generation.jpg'
    plt.savefig(plot_name, bbox_extra_artists=([leg1]), bbox_inches='tight', dpi=1200)
  elif (genOnly == False):
    plot_name = dir_figs + 'historical_synthetic_power.jpg'
    plt.savefig(plot_name, bbox_extra_artists=([leg2]), bbox_inches='tight', dpi=1200)

  return










