""" Hess2010_model.py

Created on Jun 2, 2023
@author: Shin Satoh

Description:
Hess+2010のFigure 5を再現する。
3段のパネル:
    1段目 = 衛星公転軌道における磁場強度
    2段目 = 衛星の"Torus latitude"
    3段目 = 衛星における発電量
横軸SystemIII経度に対してプロットする。

Version
1.0.0 (Jun 2, 2023)

"""


import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import copy
import B_JRM33 as BJRM
import B_equator as BEQ
from TScmap import TScmap

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
    'lines.markersize': 2,
    'lines.linewidth': 2,
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


# %% 定数
MOON = 'Europa'
RJ = 71492E+3            # JUPITER RADIUS [m]
satovalN = np.recfromtxt('data/JRM33/satellite_foot_N.txt', skip_header=3,
                         names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])
satovalS = np.recfromtxt('data/JRM33/satellite_foot_S.txt', skip_header=3,
                         names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])
if MOON == 'Io':
    r_orbit = 5.95*RJ   # ORBITAL RADIUS [m]
    R_moon = 1.82E+6    # [m]
    vcor = 57*1E+3      # [m/s]
    n0 = 2000           # [cm-3] (Hess+2010)
    mi = 32             # 原子量
    H_p = 1*RJ          # [m]
    wlonN = copy.copy(satovalN.wlon)
    FwlonN = copy.copy(satovalN.iowlon)
    FlatN = copy.copy(satovalN.iolat)
    FwlonS = copy.copy(satovalS.iowlon)
    FlatS = copy.copy(satovalS.iolat)

if MOON == 'Europa':
    r_orbit = 9.38*RJ   # ORBITAL RADIUS [m]
    R_moon = 1.56E+6    # [m]
    vcor = 100*1E+3     # [m/s]
    n0 = 110            # [cm-3] (Cassidy+2013)
    mi = 16             # 原子量
    H_p = 1.8*RJ        # [m]
    wlonN = copy.copy(satovalN.wlon)
    FwlonN = copy.copy(satovalN.euwlon)
    FlatN = copy.copy(satovalN.eulat)
    FwlonS = copy.copy(satovalS.euwlon)
    FlatS = copy.copy(satovalS.eulat)

if MOON == 'Ganymede':
    r_orbit = 15*RJ     # ORBITAL RADIUS [m]
    R_moon = 2.63E+6    # [m]
    vcor = 150*1E+3     # [m/s]
    n0 = 40             # [cm-3]
    mi = 16             # 原子量
    H_p = 1.8*RJ        # [m]
    wlonN = copy.copy(satovalN.wlon)
    FwlonN = copy.copy(satovalN.gawlon)
    FlatN = copy.copy(satovalN.galat)
    FwlonS = copy.copy(satovalS.euwlon)
    FlatS = copy.copy(satovalS.eulat)


# %% 衛星公転軌道の磁場強度
S3wlon = np.radians(np.linspace(0, 360, 50))      # [rad]
S3lat = np.zeros(S3wlon.shape)                    # [rad]

theta = 0.5*np.pi-S3lat
phi = 2*np.pi-S3wlon
B = np.zeros(phi.shape)
Br = np.zeros(phi.shape)
Btheta = np.zeros(phi.shape)
for i in range(phi.size):
    Bvec = BJRM.B().JRM33(r_orbit, theta[i], phi[i])*1E-5      # [G]
    Bvec *= 1E-4    # [T]
    B[i] = np.sqrt(Bvec[0]**2+Bvec[1]**2+Bvec[2]**2)
    Br[i] = Bvec[0]
    Btheta[i] = Bvec[1]


# %% 衛星のトーラス内相対位置
# S3RH座標系
x0 = r_orbit*np.cos(phi)
y0 = r_orbit*np.sin(phi)
z0 = np.zeros(x0.shape)
rvec = np.array([x0, y0, z0])
# print(rvec.shape)

# ダイポール座標系に持っていく
# S3RH で Z3軸 の(右ネジ)まわりに-69.2度回転
phiRH0 = math.radians(-69.2)
rvec = np.array([
    rvec[0, :]*math.cos(phiRH0) - rvec[1, :]*math.sin(phiRH0),
    rvec[0, :]*math.sin(phiRH0) + rvec[1, :]*math.cos(phiRH0),
    rvec[2, :]
])

# S3RH で X3軸 の(右ネジ)まわりに-7度回転
TILT0 = math.radians(7)
rvec = np.array([
    rvec[0, :],
    rvec[1, :]*math.cos(TILT0) - rvec[2, :]*math.sin(TILT0),
    rvec[1, :]*math.sin(TILT0) + rvec[2, :]*math.cos(TILT0)
])


# %% 発電量 Power
MU0 = 1.26E-6
AMU = 1.66E-27      # [kg]
rho0 = n0*mi*AMU    # [kg cm-3]
rho0 *= 1E+6        # [m-3]
rho = rho0*np.exp(-(rvec[2, :]/H_p)**2)
print('rho0', rho0*1E-6/AMU)

# E field
E = - vcor*(B)

# Alfven conductance
Sig_A = np.sqrt(rho/(MU0*(B**2)))

# Current [A]
J = 2*E*R_moon*Sig_A

# Power [W]
P = 4*(E**2)*(R_moon**2)*Sig_A


# %% 表面磁場強度
# 北半球
B_surfN = np.zeros(len(FwlonN))
for i in range(len(FwlonN)):
    lat = FlatN[i]
    wlong = FwlonN[i]
    # RADIUS OF SURFACE (1/15 DYNAMICALLY FLATTENED SURFACE)
    rs = RJ*np.sqrt(np.cos(np.radians(lat))**2 +
                    (np.sin(np.radians(lat))*14.4/15.4)**2)
    theta = np.radians(90-lat)
    phi = np.radians(360-wlong)
    Bvec = BJRM.B().JRM33(rs, theta, phi)*1E-5  # [G]
    Babs = math.sqrt(Bvec[0]**2+Bvec[1]**2+Bvec[2]**2)
    B_surfN[i] = Babs

# 南半球
B_surfS = np.zeros(len(FwlonS))
for i in range(len(FwlonS)):
    lat = FlatS[i]
    wlong = FwlonS[i]
    # RADIUS OF SURFACE (1/15 DYNAMICALLY FLATTENED SURFACE)
    rs = RJ*np.sqrt(np.cos(np.radians(lat))**2 +
                    (np.sin(np.radians(lat))*14.4/15.4)**2)
    theta = np.radians(90-lat)
    phi = np.radians(360-wlong)
    Bvec = BJRM.B().JRM33(rs, theta, phi)*1E-5  # [G]
    Babs = math.sqrt(Bvec[0]**2+Bvec[1]**2+Bvec[2]**2)
    B_surfS[i] = Babs


# %% ロスコーンアングル
S3wlonN = np.radians(np.linspace(0, 360, B_surfN.size))      # [rad]
S3lat = np.zeros(S3wlonN.shape)                    # [rad]
theta = 0.5*np.pi-S3lat
phi = 2*np.pi-S3wlonN
Beq = np.zeros(phi.shape)
for i in range(phi.size):
    Bvec = BJRM.B().JRM33(r_orbit, theta[i], phi[i])*1E-5      # [G]
    Beq[i] = np.sqrt(Bvec[0]**2+Bvec[1]**2+Bvec[2]**2)
alossN = np.degrees(np.arcsin(np.sqrt(Beq/B_surfN)))

S3wlonS = np.radians(np.linspace(0, 360, B_surfS.size))      # [rad]
S3lat = np.zeros(S3wlonS.shape)                    # [rad]
theta = 0.5*np.pi-S3lat
phi = 2*np.pi-S3wlonS
Beq = np.zeros(phi.shape)
for i in range(phi.size):
    Bvec = BJRM.B().JRM33(r_orbit, theta[i], phi[i])*1E-5      # [G]
    Beq[i] = np.sqrt(Bvec[0]**2+Bvec[1]**2+Bvec[2]**2)
alossS = np.degrees(np.arcsin(np.sqrt(Beq/B_surfS)))


# %% 描画
fsize = 14
fig, ax = plt.subplots(4, 1, figsize=(5.5, 7), dpi=150)
plt.subplots_adjust(left=0.2, top=0.95)
ax[0].set_title(MOON+' (Hess+2010 Model)', weight='bold', fontsize=fsize)
ax[0].set_xlim(0, 360)
# ax[0].set_ylim(1700, 2200)
# ax[0].set_xlabel('S3 wlong.')
ax[0].set_ylabel('$B_{orbit}$ [nT]', fontsize=fsize)
# ax[0].set_yticks(np.arange(1700, 2201, 100))
# ax[0].set_yticklabels(np.arange(1700, 2201, 100))
ax[0].plot(np.degrees(S3wlon), B*(1E+9), color='k')
ax[0].tick_params(axis='x', length=0, which='major')  # 目盛りを消す
plt.setp(ax[0].get_xticklabels(), visible=False)  # ラベルを消す
ax[0].text(0.01, 0.95, '(a)', color='k',
           horizontalalignment='left',
           verticalalignment='top',
           transform=ax[0].transAxes,
           fontsize=fsize)
ax[0].text(0.98, 0.95, 'Moon orbit', color='k',
           horizontalalignment='right',
           verticalalignment='top',
           transform=ax[0].transAxes,
           fontsize=fsize)

ax[1].set_xlim(0, 360)
ax[1].set_ylim(-2.0, 2.0)
ax[1].set_ylabel('Moon latitude\n$z$ [$R_J$]', fontsize=fsize)
ax[1].set_yticks(np.arange(-2, 3, 1))
ax[1].set_yticklabels(np.arange(-2, 3, 1))
ax[1].plot(np.degrees(S3wlon), rvec[2, :]/RJ, color='k')
ax[1].tick_params(axis='x', length=0, which='major')  # 目盛りを消す
plt.setp(ax[1].get_xticklabels(), visible=False)  # ラベルを消す
ax[1].text(0.01, 0.95, '(b)', color='k',
           horizontalalignment='left',
           verticalalignment='top',
           transform=ax[1].transAxes,
           fontsize=fsize)

ax[2].set_xlim(0, 360)
# ax[2].set_ylim(0.5, 1.0)
# ax[2].set_xlabel('System III longitude of the moon [deg]')
ax[2].set_ylabel('$P$ [10$^{12}$ W]', fontsize=fsize)
# ax[2].set_yticks(np.linspace(0.5, 1, 6))
# ax[2].set_yticklabels(np.linspace(0.5, 1, 6))
ax[2].plot(np.degrees(S3wlon), P*1E-12, color='k')
ax[2].tick_params(axis='x', length=0, which='major')  # 目盛りを消す
plt.setp(ax[2].get_xticklabels(), visible=False)  # ラベルを消す
ax[2].text(0.01, 0.95, '(c)', color='k',
           horizontalalignment='left',
           verticalalignment='top',
           transform=ax[2].transAxes,
           fontsize=fsize)

ax[3].set_xlim(0, 360)
# ax[3].set_ylim(0.5, 1.0)
ax[3].set_xlabel('System III longitude of the moon [deg]', fontsize=fsize)
ax[3].set_ylabel('$B_{surf}$ [G]', fontsize=fsize)
ax[3].set_xticks(np.arange(0, 361, 45))
ax[3].set_xticklabels(np.arange(0, 361, 45), fontsize=fsize)
# ax[3].set_yticks(np.linspace(0.5, 1, 6))
# ax[3].set_yticklabels(np.linspace(0.5, 1, 6))
ax[3].plot(wlonN, B_surfN, color='r')
ax[3].plot(wlonN, B_surfS, color='b')
ax[3].text(0.01, 0.95, '(d)', color='k',
           horizontalalignment='left',
           verticalalignment='top',
           transform=ax[3].transAxes,
           fontsize=fsize)
ax[3].text(0.98, 0.95, 'Surface', color='k',
           horizontalalignment='right',
           verticalalignment='top',
           transform=ax[3].transAxes,
           fontsize=fsize)
ax[3].text(0.98, 0.80, 'North', color='r',
           horizontalalignment='right',
           verticalalignment='top',
           transform=ax[3].transAxes,
           fontsize=fsize)
ax[3].text(0.98, 0.65, 'South', color='b',
           horizontalalignment='right',
           verticalalignment='top',
           transform=ax[3].transAxes,
           fontsize=fsize)
# ax[3].tick_params(axis='x', length=0, which='major')  # 目盛りを消す
# plt.setp(ax[3].get_xticklabels(), visible=False)  # ラベルを消す
ax[3].xaxis.set_minor_locator(AutoMinorLocator(3))  # minor ticks

"""
ax[4].set_xlim(0, 360)
# ax[4].set_ylim(0, 1.0)
ax[4].set_xlabel('System III longitude of the moon [deg]', fontsize=fsize)
ax[4].set_ylabel('$\\alpha_{loss}$ [deg]', fontsize=fsize)
ax[4].set_xticks(np.arange(0, 361, 45))
ax[4].set_xticklabels(np.arange(0, 361, 45), fontsize=fsize)
# ax[4].set_yticks(np.linspace(0.5, 1, 6))
# ax[4].set_yticklabels(np.linspace(0.5, 1, 6))
ax[4].plot(np.degrees(S3wlonN), alossN, color='r')
ax[4].plot(np.degrees(S3wlonS), alossS, color='b')
ax[4].text(0.01, 0.95, '(e)', color='k',
           horizontalalignment='left',
           verticalalignment='top',
           transform=ax[4].transAxes,
           fontsize=fsize)
ax[4].text(0.98, 0.95, 'North', color='r',
           horizontalalignment='right',
           verticalalignment='top',
           transform=ax[4].transAxes,
           fontsize=fsize)
ax[4].text(0.98, 0.80, 'South', color='b',
           horizontalalignment='right',
           verticalalignment='top',
           transform=ax[4].transAxes,
           fontsize=fsize)
ax[4].xaxis.set_minor_locator(AutoMinorLocator(3))  # minor ticks
"""

for i in range(4):
    ax[i].tick_params(axis='y', labelsize=fsize)

fig.tight_layout()
plt.savefig('Hess2010.png')
plt.show()

# %%
