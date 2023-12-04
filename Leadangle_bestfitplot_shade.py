""" Leadangle_bestfitplot.py

Created on Nov 16, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Nov 16, 2023)

"""
# %% Import
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import B_JRM33 as BJRM
import B_equator as BEQ
import Leadangle_wave as LeadA
from TScmap import TScmap
import time

# Color universal design
cud4 = ['#FF3300', '#FFF100', '#03AF7A', '#005AFF',
        '#4DC4FF', '#FF8082', '#F6AA00', '#990099', '#804000']
cud4bs = ['#FFCABF', '#FFFF80', '#D8F255', '#BFE4FF',
          '#FFCA80', '#77D9A8', '#C9ACE6', '#84919E']


# %% Switch
hem = 'North'       # 'North' or 'South'
year = 2022      # 2014, 2022, 202218509, 202231019, 202234923
exname = '2022_R4'
Nphi = 360          # S3経度の分割数
rho0_best = 1708.7444720361773   # [amu cm-3]
Ti_best = 149.6341687857023      # [eV]
tag = 'Lower'


# %% matplotlib フォント設定
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

Hp0 = 0.64*RJ        # H0 [m] (Bagenal&Delamere2011)

wlonN = copy.copy(satovalN.wlon)
FwlonN = copy.copy(satovalN.euwlon)
FlatN = copy.copy(satovalN.eulat)
FwlonS = copy.copy(satovalS.euwlon)
FlatS = copy.copy(satovalS.eulat)

OMGR = OMGJ-OMG_E
Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
OMGR = 2*np.pi/(Psyn_eu)    # Moon's synodic angular velocity [rad/sec]


# %% Colors
cpalette = ['#E7534B', '#EF8C46', '#F7D702', '#3C8867',
            '#649CDB', '#5341A5', '#A65FAC', '#A2AAAD']


# %% HST data
north_doy14 = ['14/006_v06', '14/013_v13', '14/016_v12']
north_doy22 = ['22/271_v18', '22/274_v17']
north_doy = north_doy14 + north_doy22
south_doy = ['22/185_v09', '22/310_v19', '22/349_v23']
south_doy = ['22/185_v09_MAW_']

nbkg_doy = ['14/001_v01', '14/002_v02', '14/005_v05', '14/006_v06', '14/013_v13', '14/016_v12',
            '22/185_v10', '22/228_v13', '22/271_v18', '22/274_v17', '22/309_v20', '22/349_v24']
sbkg_doy = ['22/140_v03', '22/185_v09', '22/186_v11',
            '22/229_v14', '22/310_v19', '22/311_v21', '22/349_v23']

refnum = 'N'
if hem == 'South':
    refnum = 'S'
satoval = np.recfromtxt('data/JRM33/satellite_foot_'+refnum+'.txt', skip_header=3,
                        names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])

if year == 2014:
    doy1422 = north_doy14
    la_err = np.array([1.0614052546455, 1.1145213015136, 0.8732066510795])

elif year == 2022:
    doy1422 = north_doy22
    la_err = np.array([1.0478863038047, 0.8994575700965])

elif year == 202218509:
    doy1422 = ['22/185_v09R']
    la_err = np.array([0.6379928352107])

elif year == 2022185:
    doy1422 = ['22/185_v09_MAW_R']
    la_err = np.array([0.6379928352107])

elif year == 202231019:
    doy1422 = ['22/310_v19R']
    la_err = np.array([0.56611511426765])

elif year == 202234923:
    doy1422 = ['22/349_v23R']
    la_err = np.array([0.6325058969635])

else:
    raise ValueError


# Set legend shadow
def legend_shadow(fig, ax, legend, dx, dy):

    frame = legend.get_window_extent()

    xmin, ymin = fig.transFigure.inverted().transform((frame.xmin, frame.ymin))
    xmax, ymax = fig.transFigure.inverted().transform((frame.xmax, frame.ymax))

    # plot patch shadow
    rect = patches.Rectangle((xmin+dx, ymin+dy), xmax-xmin, ymax-ymin,
                             transform=fig.transFigure,
                             edgecolor='k', facecolor='k',
                             clip_on=False)
    ax.add_patch(rect)

    return None


# Plot using the best-fit parameters
def BEST_FIT(rho0_best: float, HP_best: float, tag: str):

    # System III longitude of the Alfven field line [deg]
    S3_wlon = np.linspace(-25, 360, Nphi)

    eqlead_est = np.zeros(S3_wlon.size)

    for i in range(S3_wlon.size):
        S0 = LeadA.Awave().tracefield(r_orbitM, math.radians(S3_wlon[i]))
        tau, _, _ = LeadA.Awave().tracefield2(r_orbitM,
                                              math.radians(S3_wlon[i]),
                                              S0,
                                              rho0_best,
                                              HP_best,
                                              refnum)

        eqlead_est[i] = np.degrees(OMGR*tau)  # [deg]

    # System III longitude of the moon [deg]
    moonS3wlon = S3_wlon+eqlead_est

    np.savetxt('img/LeadangleFit/'+exname+'/bestfit_'+tag+'.txt',
               np.array([moonS3wlon, eqlead_est]))

    return 0


# %%
def FIT_PLOT(rho0_best: float, HP_best: float, TIME=False):

    # Plot
    fontsize = 19
    fig, ax = plt.subplots(figsize=(5, 3), dpi=144)
    ax.tick_params(axis='both', which='both', labelsize=fontsize)
    ax.set_title(str(hem)+' in '+str(year), fontsize=fontsize, weight='bold')
    ax.set_xlabel(
        "Moon System III longitude $\\lambda_{III}$ [deg]", fontsize=fontsize)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xticklabels(np.arange(0, 361, 45))
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))  # minor ticks

    savetag = 'ang'
    ax.set_ylabel('Eq. lead angle\n$\delta_{eq}$ [deg]', fontsize=fontsize)
    ax.set_ylim(-1.05, 18)      # [deg]
    ax.set_yticks(np.arange(0, 18, 5))
    ax.set_yticklabels(np.arange(0, 18, 5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # minor ticks

    # Observation
    for i in range(len(doy1422)):
        data_arr = np.loadtxt(
            'data/red3_leadangle/EUROPA/20'+doy1422[i]+'_eq.txt')
        sc = ax.scatter(data_arr[0, :], data_arr[1, :], marker=',', s=0.9, c=cud4[0],
                        linewidths=0.65, zorder=1)
        ax.errorbar(data_arr[0, :], data_arr[1, :], yerr=la_err[i]*np.ones(data_arr[0, :].shape),
                    linestyle='none', ecolor=cud4[0], elinewidth=0.22, marker='none', zorder=1)
        if i == 0:
            sc.set_label('HST')

    # Bestfit
    data_arr = np.loadtxt(
        'img/LeadangleFit/'+exname+'/bestfit.txt')
    ax.plot(data_arr[0, :], data_arr[1, :],
            linestyle='dashed', linewidth=1.3, color='k',
            label='Best fit', zorder=0.8)

    """
    # Left & Right
    data_arr = np.loadtxt(
        'img/LeadangleFit/'+exname+'/bestfit_Left.txt')
    ax.plot(data_arr[0, :], data_arr[1, :],
            linestyle='dashed', linewidth=1.3, color='purple', zorder=0.8)
    data_arr = np.loadtxt(
        'img/LeadangleFit/'+exname+'/bestfit_Right.txt')
    ax.plot(data_arr[0, :], data_arr[1, :],
            linestyle='dashed', linewidth=1.3, color='purple', zorder=0.8)

    # Upper & Lower
    data_arr = np.loadtxt(
        'img/LeadangleFit/'+exname+'/bestfit_Upper.txt')
    ax.plot(data_arr[0, :], data_arr[1, :],
            linestyle='dashed', linewidth=1.3, color='yellow', zorder=0.8)
    data_arr = np.loadtxt(
        'img/LeadangleFit/'+exname+'/bestfit_Lower.txt')
    ax.plot(data_arr[0, :], data_arr[1, :],
            linestyle='dashed', linewidth=1.3, color='yellow', zorder=0.8)
    """

    # 3-sigma borders
    s3wlon = np.linspace(0, 362, 122)
    Upper = np.zeros(s3wlon.size-1)
    Lower = np.zeros(s3wlon.size-1)
    Left = np.zeros(s3wlon.size-1)
    Right = np.zeros(s3wlon.size-1)
    for i in range(s3wlon.size-1):
        data_arr = np.loadtxt(
            'img/LeadangleFit/'+exname+'/bestfit_Upper.txt')
        cut = np.where(((data_arr[0, :] >= s3wlon[i])
                        & (data_arr[0, :] < s3wlon[i+1])))
        Upper[i] = np.average(data_arr[1, :][cut])

        data_arr = np.loadtxt(
            'img/LeadangleFit/'+exname+'/bestfit_Lower.txt')
        cut = np.where(((data_arr[0, :] >= s3wlon[i])
                        & (data_arr[0, :] < s3wlon[i+1])))
        Lower[i] = np.average(data_arr[1, :][cut])

        data_arr = np.loadtxt(
            'img/LeadangleFit/'+exname+'/bestfit_Left.txt')
        cut = np.where(((data_arr[0, :] >= s3wlon[i])
                        & (data_arr[0, :] < s3wlon[i+1])))
        Left[i] = np.average(data_arr[1, :][cut])

        data_arr = np.loadtxt(
            'img/LeadangleFit/'+exname+'/bestfit_Right.txt')
        cut = np.where(((data_arr[0, :] >= s3wlon[i])
                        & (data_arr[0, :] < s3wlon[i+1])))
        Right[i] = np.average(data_arr[1, :][cut])

    ax.fill_between(s3wlon[:-1], Left, Right,
                    facecolor='pink', alpha=1, zorder=0.6)
    ax.fill_between(s3wlon[:-1], Left, Lower,
                    facecolor='pink', alpha=1, zorder=0.6)
    ax.fill_between(s3wlon[:-1], Left, Upper,
                    facecolor='pink', alpha=1, zorder=0.6)
    ax.fill_between(s3wlon[:-1], Right, Lower,
                    facecolor='pink', alpha=1, zorder=0.6)
    ax.fill_between(s3wlon[:-1], Right, Upper,
                    facecolor='pink', alpha=1, zorder=0.6)
    ax.fill_between(s3wlon[:-1], Lower, Upper,
                    facecolor='pink', alpha=1, zorder=0.6)

    legend3 = ax.legend(loc='upper right',
                        ncol=2,
                        markerscale=5,
                        bbox_to_anchor=(1.07, 0.98),
                        fancybox=False,
                        facecolor='white',
                        framealpha=1,
                        edgecolor='k',
                        fontsize=fontsize*0.62,
                        labelspacing=0.34,
                        handlelength=1,)
    # legend1.set_title('Bagenal+2015', prop={'size': fontsize*0.62, 'weight': 'bold'})
    legend_shadow(fig, ax, legend3, dx=0.006, dy=-0.0078)

    # get plot colors
    i = 0
    txtcolor = [cud4[0], 'k', cud4[3], 'k', 'g']
    for legtext in legend3.get_texts():
        legtext.set_color(txtcolor[i])
        i += 1

    plt.savefig('img/LeadangleFit/'+exname +
                '/bestfit_'+savetag+'_shade.png', bbox_inches='tight')
    plt.close()

    return 0


# %% EXECUTE
if __name__ == '__main__':

    # Plasma sheet scale height
    HP_best = Hp0*np.sqrt(Ti_best/Ai_1)     # [m] (Bagenal&Delamere2011)

    # BEST_FIT(rho0_best, HP_best, tag=tag)
    FIT_PLOT(rho0_best, HP_best, TIME=False)

    print('done')
