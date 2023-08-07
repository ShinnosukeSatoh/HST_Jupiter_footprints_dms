""" Leadangle_fit_Juno.py

Created on Jul 21, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Jul 21, 2023)

"""

from multiprocessing import Pool
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
import matplotlib.patheffects as pe
import B_JRM33 as BJRM
import B_equator as BEQ
import Leadangle_wave as LeadA
from TScmap import TScmap
import time
import pandas as pd

# matplotlib フォント設定
fontname = 'Nimbus Sans'
plt.rcParams.update({'font.sans-serif': fontname,
                     'font.family': 'sans-serif',
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': fontname,
                     'mathtext.it': fontname+':italic',
                     # 'mathtext.bf': 'Nimbus Sans:italic:bold',
                     'mathtext.bf': fontname+':bold'
                     })
params = {
    # 'lines.markersize': 1,
    # 'lines.linewidth': 1,
    'axes.linewidth': 2,
    'xtick.major.size': 5,
    'xtick.minor.size': 3.5,
    'xtick.major.width': 2.0,
    'xtick.minor.width': 1.25,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'ytick.major.width': 2.0,
    'ytick.minor.width': 1.25,
}
plt.rcParams.update(params)


# Original colormap
def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


whiteblue = generate_cmap(
    ['#000000', '#010E5E', '#042AA6', '#0F7CE0', '#1AC7FF', '#FFFFFF'])


# %% 定数
MOON = 'Europa'
MU0 = 1.26E-6            # 真空中の透磁率
AMU = 1.66E-27           # [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
C = 2.99792E+8           # 光速 [m/s]
OMGJ = 1.75868E-4        # 木星の自転角速度 [rad/s]
satovalN = np.recfromtxt('data/JRM33/satellite_foot_N.txt', skip_header=3,
                         names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])
satovalS = np.recfromtxt('data/JRM33/satellite_foot_S.txt', skip_header=3,
                         names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])


# 定数
r_orbitM = 9.38*RJ   # ORBITAL RADIUS (average) [m] (Bagenal+2015)
r_orbitC = 9.30*RJ   # ORBITAL RADIUS (closest) [m] (Bagenal+2015)
r_orbitF = 9.47*RJ   # ORBITAL RADIUS (farthest) [m] (Bagenal+2015)
MOONRADI = 1.56E+6     # MOON RADIUS [m]
OMG_E = 2.0478E-5     # 衛星の公転角速度 [rad/s]
n0 = 110             # [cm-3] (Cassidy+2013)

ne_1 = 63            # 電子数密度 [cm-3] (Bagenal+2015)
ne_2 = 158           # 電子数密度 [cm-3] (Bagenal+2015)
ne_3 = 290           # 電子数密度 [cm-3] (Bagenal+2015)

Ai_1 = 18            # 平均イオン原子量 (Bagenal+2015)
Ai_2 = 18            # 平均イオン原子量 (Bagenal+2015)
Ai_3 = 18            # 平均イオン原子量 (Bagenal+2015)

Ti_1 = 340           # 平均イオン温度 [eV] (Bagenal+2015)
Ti_2 = 88            # 平均イオン温度 [eV] (Bagenal+2015)
Ti_3 = 48            # 平均イオン温度 [eV] (Bagenal+2015)

Zi_1 = 1.4           # 平均イオン価数 [q] (Bagenal+2015)
Zi_2 = 1.4           # 平均イオン価数 [q](Bagenal+2015)
Zi_3 = 1.4           # 平均イオン価数 [q](Bagenal+2015)

rho0_1 = 800         # プラズマ質量密度 [amu cm-3] (Bagenal+2015)
rho0_2 = 2000        # プラズマ質量密度 [amu cm-3] (Bagenal+2015)
rho0_3 = 3600        # プラズマ質量密度 [amu cm-3] (Bagenal+2015)

# H_p = 1.8*RJ         # [m]
Hp0 = 0.64*RJ        # H0 [m] (Bagenal&Delamere2011)
Hp_1 = Hp0*math.sqrt(Ti_1/Ai_1)     # Scale height [m] (Bagenal&Delamere2011)
Hp_2 = Hp0*math.sqrt(Ti_2/Ai_2)     # Scale height [m] (Bagenal&Delamere2011)
Hp_3 = Hp0*math.sqrt(Ti_3/Ai_3)     # Scale height [m] (Bagenal&Delamere2011)


wlonN = copy.copy(satovalN.wlon)
FwlonN = copy.copy(satovalN.euwlon)
FlatN = copy.copy(satovalN.eulat)
FwlonS = copy.copy(satovalS.euwlon)
FlatS = copy.copy(satovalS.eulat)

OMGR = OMGJ-OMG_E


# %% HST data
cpalette = ['#E7534B', '#EF8C46', '#F7D702', '#3C8867',
            '#649CDB', '#5341A5', '#A65FAC', '#A2AAAD']

hem = 'North'

north_doy14 = ['14/006_v06', '14/013_v13', '14/016_v12']
north_doy22 = ['22/271_v18', '22/274_v17']
north_doy = north_doy14 + north_doy22
south_doy = ['22/185_v09', '22/310_v19', '22/349_v23']

nbkg_doy = ['14/001_v01', '14/002_v02', '14/005_v05', '14/006_v06', '14/013_v13', '14/016_v12',
            '22/185_v10', '22/228_v13', '22/271_v18', '22/274_v17', '22/309_v20', '22/349_v24']
sbkg_doy = ['22/140_v03', '22/185_v09', '22/186_v11',
            '22/229_v14', '22/310_v19', '22/311_v21', '22/349_v23']

refnum = 'N'
if hem == 'South':
    refnum = 'S'
satoval = np.recfromtxt('data/JRM33/satellite_foot_'+refnum+'.txt', skip_header=3,
                        names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])


# %% Function to be in loop
def calcN(RHO0: float, HP: float, MOONS3: float):
    """
    `RHO0`: plasma density at the equator [amu cm-3] \\
    `HP`: scale height of the plasma sheet [m] \\
    `moons3`: system iii longitude of the moon [deg]
    """

    S0 = LeadA.Awave().tracefield(r_orbitM, math.radians(MOONS3))
    tau, _, _ = LeadA.Awave().tracefield2(r_orbitM,
                                          math.radians(MOONS3),
                                          S0,
                                          RHO0,
                                          HP,
                                          'N')
    return tau


def calcS(RHO0: float, HP: float, MOONS3: float):
    """
    `RHO0`: plasma density at the equator [amu cm-3] \\
    `HP`: scale height of the plasma sheet [m] \\
    `moons3`: system iii longitude of the moon [deg]
    """

    S0 = LeadA.Awave().tracefield(r_orbitM, math.radians(MOONS3))
    tau, _, _ = LeadA.Awave().tracefield2(r_orbitM,
                                          math.radians(MOONS3),
                                          S0,
                                          RHO0,
                                          HP,
                                          'S')
    return tau


# %% Plot function
def LAplot(estimations, RHO0: float, TI: float, chi2: float, II: int, JJ: int):
    """
    `estimations`: estimations of equatorial lead angle \\
    ---`estimations[0,:]`: satellite system iii longitude [deg] \\
    ---`estimations[1,:]`: equatorial lead angle of actual auroral spot [deg] \\
    ---`estimations[2,:]`: model estimation of equatorial lead angle of auroral spot [deg] \\
    `RHO0`: plasma density at the equator [amu cm-3] \\
    `TI`: plasma temperature [eV] \\
    `chi2`: chi square value \\
    `II`: index \\
    `JJ`: index
    """
    fontsize = 15
    fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=100)
    ax.set_title('('+str(II)+', '+str(JJ)+') Equatorial Lead Angle $\\delta$',
                 weight='bold', fontsize=fontsize)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xticklabels(np.arange(0, 361, 45), fontsize=fontsize)
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))  # minor ticks
    ax.set_yticks(np.arange(0, 21, 5))
    ax.set_yticklabels(np.arange(0, 21, 5), fontsize=fontsize)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # minor ticks
    ax.set_xlabel('Europa S3 longitude [deg]', fontsize=fontsize)
    ax.set_ylabel('$\\delta$ [deg]', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    ax.scatter(estimations[0, :], estimations[1, :], marker='x',
               c='r', linewidths=0.5, zorder=1)
    ax.scatter(estimations[0, :], estimations[2, :],
               s=1, color='k', zorder=2)

    ax.text(0.98, 0.96, '$\\rho_{\,0}=$'+str(round(RHO0, 1))+' amu cm$^{-3}$', color='k',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=fontsize*0.72)
    ax.text(0.98, 0.79, '$\\langle T_i \, \\rangle=$'+str(round(TI, 1))+' eV', color='k',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=fontsize*0.72)
    ax.text(0.98, 0.58, '$\\chi^2=$'+str(round(chi2, 1)), color='k',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=fontsize*0.72)

    fig.tight_layout()
    plt.savefig('img/LeadangleFit/PJ32/img/fit_'+str(II)+'_' +
                str(JJ)+'.png', bbox_inches='tight')
    plt.close()
    return 0


def main(RHO0: float, Ti0: float, HP0: float, II: int, JJ: int, year: int):
    """
    `RHO0`: plasma density at the equator [amu cm-3] \\
    `Ti0`: plasma temperature [eV] \\
    `HP0`: scale height of the plasma sheet [m]
    """

    PJnum = 'PJ32'
    data_pd = pd.read_csv('data/Hue23_EFP_north2.txt', skiprows=2,
                          names=['PJ', 'UTC_TIME',
                                 'Moon_SIII_LON',
                                 'FP_LON',
                                 'FP_LAT',
                                 'FP_LON_ERR',
                                 'FP_LAT_ERR',
                                 'EMISSION_ANGLE',
                                 'EQ_LEAD_ANGLE',
                                 'EQ_LEAD_ANGLE_ERR'],
                          sep=',')
    PJdata = data_pd[data_pd['PJ'].str.contains(PJnum)]
    ests = np.zeros((4, PJdata.shape[0]))
    ests[0, :] = PJdata['Moon_SIII_LON']  # moon s3 longitude [deg]
    ests[1, :] = PJdata['EQ_LEAD_ANGLE']  # observed lead angle [deg]
    ests[3, :] = PJdata['EQ_LEAD_ANGLE_ERR']  # error [deg]

    # 並列計算用 変数リスト(zip)
    # np.arrayは不可。ここが1次元なのでpoolの結果も1次元。
    args = list(zip(
        RHO0*np.ones(ests.shape[1]),
        HP0*np.ones(ests.shape[1]),
        ests[0, :]))

    start = time.time()
    with Pool(processes=17) as pool:
        result_list = list(pool.starmap(calcN, args))
    tau = np.array(result_list)         # [sec]
    print(str(II)+' '+str(JJ)+' time', time.time()-start)

    leadangle_est = np.degrees(OMGR*tau)  # [deg]

    # estimated lead angle [deg]
    ests[2, :] = leadangle_est

    # CHI SQUARE VALUE
    chi2 = np.sum(((ests[1, :]-ests[2, :])/ests[3, :])**2)

    # PLOT
    LAplot(ests, RHO0, Ti0, chi2, II, JJ)

    return chi2


def main2(RHO0: float, Ti0: float, HP0: float, II: int, JJ: int, year: int):
    """
    `RHO0`: plasma density at the equator [amu cm-3] \\
    `Ti0`: plasma temperature [eV] \\
    `HP0`: scale height of the plasma sheet [m]
    """

    PJnum = 'PJ32'

    # Northern hemisphere
    data_pd = pd.read_csv('data/Hue23_EFP_north2.txt', skiprows=2,
                          names=['PJ', 'UTC_TIME',
                                 'Moon_SIII_LON',
                                 'FP_LON',
                                 'FP_LAT',
                                 'FP_LON_ERR',
                                 'FP_LAT_ERR',
                                 'EMISSION_ANGLE',
                                 'EQ_LEAD_ANGLE',
                                 'EQ_LEAD_ANGLE_ERR'],
                          sep=',')
    PJdata = data_pd[data_pd['PJ'].str.contains(PJnum)]
    estsN = np.zeros((4, PJdata.shape[0]))
    estsN[0, :] = PJdata['Moon_SIII_LON']  # moon s3 longitude [deg]
    estsN[1, :] = PJdata['EQ_LEAD_ANGLE']  # observed lead angle [deg]
    estsN[3, :] = PJdata['EQ_LEAD_ANGLE_ERR']  # error [deg]

    # 並列計算用 変数リスト(zip)
    # np.arrayは不可。ここが1次元なのでpoolの結果も1次元。
    args = list(zip(
        RHO0*np.ones(estsN.shape[1]),
        HP0*np.ones(estsN.shape[1]),
        estsN[0, :]))

    start = time.time()
    with Pool(processes=10) as pool:
        result_list = list(pool.starmap(calcN, args))
    tau = np.array(result_list)         # [sec]
    print(str(II)+' '+str(JJ)+' time', time.time()-start)

    # estimated lead angle [deg]
    estsN[2, :] = np.degrees(OMGR*tau)  # [deg]

    # CHI SQUARE VALUE
    chi2 = np.sum(((estsN[1, :]-estsN[2, :])/estsN[3, :])**2)

    # Southern hemisphere
    data_pd = pd.read_csv('data/Hue23_EFP_south2.txt', skiprows=2,
                          names=['PJ', 'UTC_TIME',
                                 'Moon_SIII_LON',
                                 'FP_LON',
                                 'FP_LAT',
                                 'FP_LON_ERR',
                                 'FP_LAT_ERR',
                                 'EMISSION_ANGLE',
                                 'EQ_LEAD_ANGLE',
                                 'EQ_LEAD_ANGLE_ERR'],
                          sep=',')
    PJdata = data_pd[data_pd['PJ'].str.contains(PJnum)]
    estsS = np.zeros((4, PJdata.shape[0]))
    estsS[0, :] = PJdata['Moon_SIII_LON']  # moon s3 longitude [deg]
    estsS[1, :] = PJdata['EQ_LEAD_ANGLE']  # observed lead angle [deg]
    estsS[3, :] = PJdata['EQ_LEAD_ANGLE_ERR']  # error [deg]

    # 並列計算用 変数リスト(zip)
    # np.arrayは不可。ここが1次元なのでpoolの結果も1次元。
    args = list(zip(
        RHO0*np.ones(estsS.shape[1]),
        HP0*np.ones(estsS.shape[1]),
        estsS[0, :]))

    start = time.time()
    with Pool(processes=13) as pool:
        result_list = list(pool.starmap(calcS, args))
    tau = np.array(result_list)         # [sec]
    print(str(II)+' '+str(JJ)+' time', time.time()-start)

    # estimated lead angle [deg]
    estsS[2, :] = np.degrees(OMGR*tau)  # [deg]

    # CHI SQUARE VALUE
    chi2 += np.sum(((estsS[1, :]-estsS[2, :])/estsS[3, :])**2)

    # PLOT
    LAplot(estsN, RHO0, Ti0, chi2, II, JJ)

    return chi2


# %% EXECUTE
if __name__ == '__main__':
    year = 0

    # Parameters
    RHO0_len = 70
    Ti0_len = 45
    RHO0 = np.linspace(np.log(300), np.log(6000), RHO0_len)
    RHO0 = np.exp(RHO0)

    Ti0 = np.linspace(np.log(20), np.log(340), Ti0_len)
    Ti0 = np.exp(Ti0)
    RHO0, Ti0 = np.meshgrid(RHO0, Ti0)

    # Plasma sheet scale height
    HP_arr = Hp0*np.sqrt(Ti0/Ai_1)     # [m] (Bagenal&Delamere2011)

    # chi2 data array
    chi2_arr = np.zeros((Ti0_len, RHO0_len))
    i = 0
    j = 0
    start = time.time()
    for i in range(Ti0_len):
        print('Hp [RJ]', HP_arr[i, 0]/RJ)
        if HP_arr[i, 0] > 2.8*RJ:
            0
            # break
        for j in range(RHO0_len):
            chi2_arr[i, j] = main2(RHO0[i, j], Ti0[i, j],
                                   HP_arr[i, j],
                                   i, j, year)
    print('total time', time.time()-start)

    print(chi2_arr)

    np.savetxt('img/LeadangleFit/PJ32/params_RHO0.txt',
               RHO0)
    np.savetxt('img/LeadangleFit/PJ32/params_Ti0.txt',
               Ti0)
    np.savetxt('img/LeadangleFit/PJ32/params_chi2.txt',
               chi2_arr)
