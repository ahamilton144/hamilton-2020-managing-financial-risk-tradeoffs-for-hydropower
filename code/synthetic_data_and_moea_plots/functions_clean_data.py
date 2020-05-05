##############################################################################################################
### functions_clean_data.py - python functions used in getting and cleaning raw data
### Project started May 2017, last update Jan 2020
##############################################################################################################

import numpy as np
import pandas as pd


##########################################################################
######### function for getting hist SWE from CDEC and cleaning ###########
############## Returns dataframe of Feb & Apr SWE (inch) #########################################
##########################################################################

def get_clean_swe(dir_downloaded_inputs):
  # ### Data originally retrieved by querying CDEC for Dana Meadows snow station
  # swe = pd.read_csv(
  #   "http://cdec.water.ca.gov/cgi-progs/snowQuery?course_num=dan&month=%28All%29&start_date=&end_date=&csv_mode=Y&data_wish=Raw+Data",
  #   delimiter='\t')
  # swe.to_csv(dir_downloaded_inputs + 'swe_dana_meadows.csv', index=False)

  # read in swe data from Dana Meadows (data originally from query above)
  swe = pd.read_csv(dir_downloaded_inputs + 'swe_dana_meadows.csv')
  # get Feb and Apr measurements, which are concurrent for years 1952-2018, except 1963 (66 years)
  swe2 = swe.loc[(swe['Month'] == 4) & (swe['Year'] > 1951) & (swe['Year'] < 2017), ['Year', 'SWE']]
  swe = swe.loc[(swe['Month'] == 2) & (swe['Year'] > 1951) & (swe['Year'] < 2017), ['Year', 'SWE']]
  swe.index = swe['Year']
  del swe['Year']
  swe.columns = ['danFeb']
  swe2.index = swe2['Year']
  swe2 = swe2.loc[swe.index]
  swe['danApr'] = swe2['SWE']
  swe['avg'] = (swe['danFeb'] + swe['danApr']) / 2

  return swe




##########################################################################
######### function for getting hist hydro generation. Clean and aggregate ###########
############## Returns dataframe of monthly generation (GWh/mnth) #########################################
##########################################################################

def get_historical_generation(dir_downloaded_inputs, swe):
  ## read in data from SFPUC. Note: SFPUC_genMonthly.csv compiled by hand from SFPUC_Combined_Public.xlsx.
  gen = pd.read_csv(dir_downloaded_inputs + '/SFPUC_genMonthly.csv')
  gen['wyear'] = gen.wyr.copy()
  del gen['wyr']
  gen['tot'] = gen.gen / 1e6
  del gen['gen']
  gen['partWYr'] = gen['wyear'] + (gen['wmnth'] - 1) / 12
  # add corresponding sweFeb and sweApr to each month in generation df
  gen['sweApr'] = np.nan
  gen['sweFeb'] = np.nan
  # cut to use only full water years
  gen = gen.loc[(gen.wyear > 1987) & (gen.wyear < 2017)]
  for i in range(0, gen.shape[0]):
    gen.sweFeb.iloc[i] = swe.loc[gen.wyear.iloc[i], 'danFeb']
    gen.sweApr.iloc[i] = swe.loc[gen.wyear.iloc[i], 'danApr']

  return gen







##########################################################################
######### function for getting hist wholesale power prices. Clean and aggregate ###########
############## Returns dataframe of monthly avg prices ($/MWh)#########################################
##########################################################################

def get_historical_power(dir_downloaded_inputs):
  # import power price data (EIA, NP-15, $/MWh)
  pow = pd.read_excel(dir_downloaded_inputs + 'NP15Hub.xls', skiprows=0)
  pow.index = pd.to_datetime(pow['Trade Date'])
  pow = pow[['Wtd Avg Price $/MWh']]
  pow.columns = ['price']

  pow2014 = pd.read_excel(dir_downloaded_inputs + 'ice_electric-2014final.xls')
  pow2014 = pow2014.loc[
    (pow2014['Price hub'] == 'NP15 EZ Gen DA LMP Peak') | (pow2014['Price hub'] == 'NP 15 EZ Gen DA LMP Peak')]
  pow2014.index = pd.to_datetime(pow2014['Trade date'])
  pow2014 = pow2014[['Wtd avg price $/MWh']]
  pow2014.columns = ['price']
  pow = pow.append(pow2014)
  del pow2014

  pow2015 = pd.read_excel(dir_downloaded_inputs + 'ice_electric-2015final.xls')
  pow2015 = pow2015.loc[
    (pow2015['Price hub'] == 'NP15 EZ Gen DA LMP Peak') | (pow2015['Price hub'] == 'NP 15 EZ Gen DA LMP Peak')]
  pow2015.index = pd.to_datetime(pow2015['Trade date'])
  pow2015 = pow2015[['Wtd avg price $/MWh']]
  pow2015.columns = ['price']
  pow = pow.append(pow2015)
  del pow2015

  pow2016 = pd.read_excel(dir_downloaded_inputs + 'ice_electric-2016final.xls')
  pow2016 = pow2016.loc[
    (pow2016['Price hub'] == 'NP15 EZ Gen DA LMP Peak') | (pow2016['Price hub'] == 'NP 15 EZ Gen DA LMP Peak')]
  pow2016.index = pd.to_datetime(pow2016['Trade date'])
  pow2016 = pow2016[['Wtd avg price $/MWh']]
  pow2016.columns = ['price']
  pow = pow.append(pow2016)
  del pow2016

  # get week of year (isocalendar)
  pow['date'] = pow.index
  # filter for wyr 2010-2016, period of relative stability in gas and power markets
  pow['wyr'] = pow.index.year
  pow['wmnth'] = pow.index.month + 3
  pow.wmnth.loc[pow.wmnth > 12] = pow.wmnth.loc[pow.wmnth > 12] - 12
  pow.wyr.loc[pow.wmnth < 4] = pow.wyr.loc[pow.wmnth < 4] + 1
  pow = pow.loc[(pow.wyr > 2009) & (pow.wyr < 2017),]

  # sample monthly
  powmnth = pd.DataFrame({'priceMean': pow.resample('m').mean().price,
                          'wyr': pow.resample('m').mean().wyr, 'wmnth': pow.resample('m').mean().wmnth})

  # adjust for inflation, calc as Oct 2016 $
  cpi = pd.read_excel(dir_downloaded_inputs + 'SeriesReport-20190311141838_d27dd7.xlsx', 'Sheet1')
  base = cpi.Oct.loc[cpi.Year == 2016].values[0]
  for i in range(powmnth.shape[0]):
    powmnth.priceMean.iloc[i] *= base / \
                                 cpi.iloc[:, powmnth.index.month[i]].loc[cpi.Year == powmnth.index.year[i]].values

  return (powmnth)




##########################################################################
######### function for getting info on SFPUC power productions, sales, and purchases, taken from financial statements ###########
############## Returns dataframes of power volumes (GWh/FY), sales ($M/FY), and rates ($/kWh or $M/GWh)#########################################
##########################################################################

def get_historical_SFPUC_sales():
  # Hetchy Power data (MWh) (SFPUC FY2014 pg 232)
  hp_GWh = pd.DataFrame({'gfrs': [0,0,0,354027, 357864, 368362, 389353, 394673, 393030, 387645, 378503, 365234, 367904, 359519, 373114],
                         'er': [813872, 829717, 851455, 482650, 487705, 489853, 495227, 484342, 471034, 481921, 485204, 484628, 493254, 487869, 495272],
                         'wspp': [370772, 139029, 212259, 158127, 368045, 36093, 125528, 217792, 298549, 568157, 143675, 131200, 2400, 0, 3040],
                         'mtid': [871807, 803593, 834549, 965348, 1004856, 548459, 386568, 258268, 286908, 459320, 277838, 227544, 103489, 115026, 377981],
                         'nca': [73710, 76085, 73425, 84788, 86326, 83378, 79351, 79231, 84378, 87142, 101128, 116996, 117289, 101605, 99568],
                         'mcr': [9310, 9459, 10011, 10660, 11681, 13211, 15556, 15094, 9578, 7652, 7552, 7808, 9206, 25472, 30451],
                         'purch': [547322, 389580, 498926, 456277, 420807, 66200, 126250, 0, 132000, 16252, 125033, 38702, 76905, 45465, 113154],
                         'gen': [1729416, 1597019, 1611949, 1728843, 1947747, 1353735, 1414703, 1522109, 1447863, 1988582, 1332957, 1312446, 1032589, 988649, 1532068],
                         'bw': [137267, 128716, 129176, 128714, 51109, -120719, 47850, 68071, -11318, -12527, -21978, 6707, 17102, 78391, 0]},
                        index = range(2002,2017))/1000

  hp_GWh['W'] = -hp_GWh['purch'] + hp_GWh['wspp']
  hp_GWh['M'] = hp_GWh['gfrs'] + hp_GWh['er'] + hp_GWh['nca'] + hp_GWh['mcr']
  hp_GWh['E'] = hp_GWh['gen'] - hp_GWh['M'] - hp_GWh['mtid'] - hp_GWh['W']

  # plt.plot(pd.DataFrame({'G': hp_GWh['gen'], 'M': hp_GWh['M'], 'I': hp_GWh['mtid'], 'W': hp_GWh['W'], 'E': hp_GWh['E']}, index=hp_GWh.index).loc[:,['G','M','I','W','E']]/1000)
  # plt.legend(['G','M','I','W','E'])
  # plt.xlabel('Financial year')
  # plt.ylabel('Generation, purchases, sales (GWh)')
  # plt.show()

  # Hetchy Power data ($) (SFPUC FY2014 pg 232)
  hp_dolM = pd.DataFrame({'gfrs': [0,0,0,11443, 11957, 11148, 12690, 13124, 14081, 13858, 13595, 13330, 15006, 18125, 22151],
                          'er': [0,0,0,43911, 47742, 48057, 48125, 50529, 50016, 52483, 52512, 52955, 60766, 65022, 65897],
                          'wspp': [0,0,0,7399, 23383, 1911, 9247, 6162, 10106, 16292, 3817, 5143, 127, 0, 50],
                          'mtid': [0,0,0,25022, 24527, 14264, 10463, 5039, 7530, 10566, 7340, 6538, 3431, 4488, 13634],
                          'nca': [0,0,0,10536, 10647, 10863, 10302, 10680, 11535, 12143, 13810, 14815, 16305, 14628, 15610],
                          'mcr': [0,0,0,602, 696, 689, 854, 854, 534, 449, 457, 486, 607, 1100, 1095],
                          'purch': [0,0,0,31117, 29031, 4158, 9004, 0, 328, 1233, 4754, 2494, 4408, 2045, 5509]},
                         index = range(2002,2017))/1000


  hp_dolM['W'] = -hp_dolM['purch'] + hp_dolM['wspp']
  hp_dolM['M'] = hp_dolM['gfrs'] + hp_dolM['er'] + hp_dolM['nca'] + hp_dolM['mcr']
  hp_dolM['R0'] = hp_dolM['M'] + hp_dolM['W'] + hp_dolM['mtid']

  # plt.plot(pd.DataFrame({'R0': hp_dolM['R0'].loc[2005:2016], 'M*rM': hp_dolM['M'].loc[2005:2016], 'I*rI': hp_dolM['mtid'].loc[2005:2016], 'W*P\'': hp_dolM['W'].loc[2005:2016]}, index=hp_GWh.index).loc[:,['R0','M*rM','I*rI','W*P\'']]/1000)
  # plt.legend(['Total','Municipal','Irrigation District','Net Wholesale'])
  # plt.xlabel('Financial year')
  # plt.ylabel('Revenue ($M)')
  # plt.show()

  # sbn.set_context('paper', font_scale=1.55)
  # sbn.regplot(x='swe', y='gen', data=pd.DataFrame({'gen': genWY['tot']/1000, 'swe': swe['danFeb'].loc[2002:2016]*32/40.7+swe['danFeb'].loc[2002:2016]*8.7/40.7}))
  # plt.xlabel('Weighted SWE (in)')
  # plt.ylabel('Hydropower generation (GWh/yr)')
  # plt.title('Hetch Hetchy Snowpack Index')

  # Hetchy Power data ($/kWh aka $M/GWh avg) (SFPUC FY2014 pg 232)
  hp_dolPerKwh = pd.DataFrame({'gfrs': hp_dolM['gfrs'] / hp_GWh['gfrs'],
                               'er': hp_dolM['er'] / hp_GWh['er'],
                               'wspp': hp_dolM['wspp'] / hp_GWh['wspp'],
                               'mtid': hp_dolM['mtid'] / hp_GWh['mtid'],
                               'nca': hp_dolM['nca'] / hp_GWh['nca'],
                               'mcr': hp_dolM['mcr'] / hp_GWh['mcr'],
                               'purch': hp_dolM['purch'] / hp_GWh['purch']},
                              index = range(2002, 2017))
  hp_dolPerKwh['M'] = hp_dolM['M'] / hp_GWh['M']

  return (hp_GWh, hp_dolPerKwh, hp_dolM)
