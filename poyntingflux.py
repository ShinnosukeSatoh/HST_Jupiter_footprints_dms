""" poyntingflux.py

Created on Apr 18, 2024
@author: Shin Satoh

Description:
This class is written specifically for calculating and locating
the magnetic equator of the Jovian magnetosphere.


Version
1.0.0 (Apr 18, 2024)

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import B_JRM33 as BJRM
import B_equator as BEQ
import Leadangle_wave as LeadA
from TScmap import TScmap

# Color universal design
cud4 = ['#FF3300', '#FFF100', '#03AF7A', '#005AFF',
        '#4DC4FF', '#FF8082', '#F6AA00', '#990099', '#804000']
cud4bs = ['#FFCABF', '#FFFF80', '#D8F255', '#BFE4FF',
          '#FFCA80', '#77D9A8', '#C9ACE6', '#84919E']

# matplotlib フォント設定
fontname = 'Nimbus Sans'
plt.rcParams.update({'font.sans-serif': fontname,
                     'font.family': 'sans-serif',
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': fontname,
                     'mathtext.it': fontname+':italic',
                     # 'mathtext.bf': 'Nimbus Sans:italic:bold',
                     'mathtext.bf': fontname+':bold',
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


# %% 定数
MOON = 'Europa'
MU0 = 1.26E-6            # 真空中の透磁率
AMU = 1.66E-27           # [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
RE = 1.56E+6             # MOON RADIUS [m]
C = 2.99792E+8           # 光速 [m/s]
OMGJ = 1.75868E-4        # 木星の自転角速度 [rad/s]
r_orbit = 9.38*RJ        # ORBITAL RADIUS (average) [m] (Bagenal+2015)
Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
OMGR = 2*np.pi/(Psyn_eu)    # Moon's synodic angular velocity [rad/sec]
u = 100E+3              # 共回転速度 [m s-1]
a_bar = 0.8             # 1-プラズマフローの減速率 (Saur+1998)
R_eff = 4.2*RE/2        # フラックスチューブの半径
Ai = 18                 # イオン原子量 [amu]
Ti = 195                # イオン温度 [eV]
rho = 1708              # プラズマシート質量密度 [amu cm-3]
rho *= AMU*1E+6         # プラズマシート質量密度 [kg m-3]


# %% 衛星公転軌道の磁場強度
S3wlon = np.radians(np.linspace(0, 360, 100))      # [rad]
S3lat = np.zeros(S3wlon.shape)                    # [rad]

theta_arr = 0.5*np.pi-S3lat
phi_arr = 2*np.pi-S3wlon

B = np.zeros(S3wlon.shape)
Br = np.zeros(S3wlon.shape)
Btheta = np.zeros(S3wlon.shape)
Bphi = np.zeros(S3wlon.shape)

x_arr = r_orbit*np.sin(theta_arr)*np.cos(phi_arr)
y_arr = r_orbit*np.sin(theta_arr)*np.sin(phi_arr)
z_arr = r_orbit*np.cos(theta_arr)

for i in range(S3wlon.size):
    theta = theta_arr[i]
    phi = phi_arr[i]
    x = x_arr[i]
    y = y_arr[i]
    z = z_arr[i]

    Bv = BJRM.B().JRM33(r_orbit, theta, phi)*1E-9        # [T]
    Bx = Bv[0]*math.sin(theta)*math.cos(phi) \
        + Bv[1]*math.cos(theta)*math.cos(phi) \
        - Bv[2]*math.sin(phi)
    By = Bv[0]*math.sin(theta)*math.sin(phi) \
        + Bv[1]*math.cos(theta)*math.sin(phi) \
        + Bv[2]*math.cos(phi)
    Bz = Bv[0]*math.cos(theta) - Bv[1]*math.sin(theta)

    Bcs = BJRM.B().BCS(x, y, z, phi)*1E-9  # [T]
    Bx += Bcs[0]
    By += Bcs[1]
    Bz += Bcs[2]

    B0 = math.sqrt(Bx**2+By**2+Bz**2)      # [T]
    Br[i] = Bx*math.sin(theta)*math.cos(phi) \
        + By*math.sin(theta)*math.sin(phi) \
        + Bz*math.cos(theta)
    Btheta[i] = Bx*math.cos(theta)*math.cos(phi) \
        + By*math.cos(theta)*math.sin(phi) \
        - Bz*math.sin(theta)
    Bphi[i] = -Bx*math.sin(phi) + By*math.cos(phi)

    B[i] = B0


# 質量密度の計算
# ダイポール座標系に持っていく
# S3RH で Z3軸 の(右ネジ)まわりに-155.8度回転
phiRH0 = math.radians(-155.8)    # Connerney+2020
rvec1 = np.array([
    x_arr*math.cos(phiRH0) - y_arr*math.sin(phiRH0),
    x_arr*math.sin(phiRH0) + y_arr*math.cos(phiRH0),
    z_arr
])

# S3RH で Y3軸 の(右ネジ)まわりに-9.3度回転
THETA_D = math.radians(-6.7)
rvec1 = np.array([
    rvec1[0]*math.cos(THETA_D) + rvec1[2]*math.sin(THETA_D),
    rvec1[1],
    -rvec1[0]*math.sin(THETA_D) + rvec1[2]*math.cos(THETA_D)
])

# 質量密度
H = 2.0*RJ
H0 = 0.64*RJ
H = H0*np.sqrt(Ti/Ai)
rho1 = rho*np.exp(-(rvec1[2]/H)**2)

# Poynting flux
S = 2*np.pi*(R_eff**2)*((a_bar*u)**2)*B*np.sqrt(rho1/MU0)

# Plot
fig, ax = plt.subplots(dpi=200)
fsize = 19
ax.tick_params(axis='both', labelsize=fsize)
ax.set_title('Poynting flux', fontsize=fsize, weight='bold')
ax.set_xlabel('System III longitude [deg]', fontsize=fsize)
ax.set_ylabel('Poynting flux [GW]', fontsize=fsize)
ax.plot(np.degrees(S3wlon), S*1E-9)

fig.tight_layout()
fig.savefig('img/Pynt.png')
plt.show()

np.savetxt('data/Poyntingflux/PY_2022_R4.txt',
           np.array([np.degrees(S3wlon), S]))


# 等高線のふち
"""
14 ===
Rho
[1303.60557713  920.52383102  788.63221675 1206.60730149  852.02978201
 1408.40147298 1581.60074835 1206.60730149  788.63221675 1581.60074835
 1116.82644318 1160.84923261]
Ti
[ 71.54245234 130.84708648 139.92567679  93.56151816 122.3575289
  81.81454916  62.56005246 100.05311621 149.63416879  66.90066944
  87.49110484 106.99512214]
22 ===
Rho
[1581.60074835 1643.94389677 1581.60074835 1994.51654932 1846.10918638
 1521.621834   1463.91749505 1846.10918638 1354.99078042 1463.91749505
 1846.10918638 1846.10918638]
Ti
[195.68800819 182.99147318 223.78497859 149.63416879 182.99147318
 209.26546949 273.67240544 195.68800819 273.67240544 255.91612437
 160.01626708 160.01626708]
"""

rho14_arr = np.array([1303.60557713, 920.52383102, 788.63221675, 1206.60730149, 852.02978201,
                      1408.40147298, 1581.60074835, 1206.60730149, 788.63221675, 1581.60074835, 1116.82644318, 1160.84923261])
rho14_arr *= AMU*1E+6
Ti14_arr = np.array([71.54245234, 130.84708648, 139.92567679, 93.56151816, 122.3575289,
                     81.81454916, 62.56005246, 100.05311621, 149.63416879, 66.90066944,
                     87.49110484, 106.99512214])

rho22_arr = np.array([1581.60074835, 1643.94389677, 1581.60074835, 1994.51654932, 1846.10918638,
                      1521.621834, 1463.91749505, 1846.10918638, 1354.99078042, 1463.91749505,
                      1846.10918638, 1846.10918638])
rho22_arr *= AMU*1E+6
Ti22_arr = np.array([195.68800819, 182.99147318, 223.78497859, 149.63416879, 182.99147318,
                     209.26546949, 273.67240544, 195.68800819, 273.67240544, 255.91612437,
                     160.01626708, 160.01626708])

for i in range(rho14_arr.size):
    # 質量密度
    rho0_14 = rho14_arr[i]
    rho0_22 = rho22_arr[i]
    Ti14 = Ti14_arr[i]
    Ti22 = Ti22_arr[i]

    H14 = H0*np.sqrt(Ti14/Ai)
    H22 = H0*np.sqrt(Ti22/Ai)
    rho14 = rho0_14*np.exp(-(rvec1[2]/H14)**2)
    rho22 = rho0_22*np.exp(-(rvec1[2]/H22)**2)

    # Poynting flux
    S14 = 2*np.pi*(R_eff**2)*((a_bar*u)**2)*B*np.sqrt(rho14/MU0)
    S22 = 2*np.pi*(R_eff**2)*((a_bar*u)**2)*B*np.sqrt(rho22/MU0)

    np.savetxt('data/Poyntingflux/PY_2014_R4_edge_'+str(i) +
               '.txt', np.array([np.degrees(S3wlon), S14]))
    np.savetxt('data/Poyntingflux/PY_2022_R4_edge_'+str(i) +
               '.txt', np.array([np.degrees(S3wlon), S22]))
