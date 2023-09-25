""" B_dip_angle.py

Created on Sep 11, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Sep 11, 2023)

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
RJ = 71492E+3            # JUPITER RADIUS [m]


# %% TOGGLES
JRM_NUM = 'JRM33'                           # 'JRM09' or 'JRM33'
SATELLITE = ['AM', 'IO', 'EU', 'GA']              # 'AM', 'IO', 'EU', 'GA'


# %% DATA LOADING FOR COEFFICIENTS AND POSITION OF SATELLITE FOOTPRINTS
jrm_coef = np.loadtxt(
    'data/'+JRM_NUM+'/coef.txt', skiprows=1, usecols=1
)
satellite_Nftp = np.loadtxt(
    'data/'+JRM_NUM+'/satellite_foot_N.txt', skiprows=3
)
satellite_Sftp = np.loadtxt(
    'data/'+JRM_NUM+'/satellite_foot_S.txt', skiprows=3
)
satellite_name = ['AM', 'IO', 'EU', 'GA']
satellite_color = ['#888888', '#000000', '#f24875', '#888888']
satellite_marker = ['o', '*', 's', 'd']
satellite_index = [satellite_name.index(i) for i in (SATELLITE)]


def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))

    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list)
    return cmap


def main():
    # B FIELD MAPPING
    lat_arr = np.linspace(-89.85, 89.85, 200)
    wlong_arr = np.linspace(0, 360, lat_arr.size*2)
    x, y = np.meshgrid(wlong_arr, lat_arr)

    Babs_arr = np.zeros(x.shape)
    Dip_arr = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            wlong = x[i, j]
            lat = y[i, j]
            rs = RJ*np.sqrt(np.cos(np.radians(lat))**2 +
                            (np.sin(np.radians(lat))*14.4/15.4)**2)
            theta = np.radians(90-lat)
            phi = np.radians(360-wlong)
            Bv = BJRM.B().JRM33(rs, theta, phi)*1E-5  # [G]

            Babs_arr[i, j] = np.sqrt(Bv[0]**2 + Bv[1]**2 + Bv[2]**2)

            # x-y-z coordinate
            Bx = Bv[0]*math.sin(theta)*math.cos(phi) + Bv[1] * \
                math.cos(theta)*math.cos(phi) - Bv[2]*math.sin(phi)
            By = Bv[0]*math.sin(theta)*math.sin(phi) + Bv[1] * \
                math.cos(theta)*math.sin(phi) + Bv[2]*math.cos(phi)
            Bz = Bv[0]*math.cos(theta) - Bv[1]*math.sin(theta)

            # Surface normal
            r0 = np.array([
                math.sin(theta)*math.cos(phi),
                math.sin(theta)*math.sin(phi),
                math.cos(theta)
            ])

            # Dip angle
            dot = np.abs(Bx*r0[0] + By*r0[1] + Bz*r0[2])
            Dip_arr[i, j] = math.degrees(math.acos(dot/Babs_arr[i, j]))


# PLOT
    """
    cmapBR = generate_cmap(
        ['#6579FE', '#819AFE', '#A0BBFF', '#BFDDFF', '#DCFFFF',
            '#FFFFDB', '#FFDEBF', '#FEBBA3', '#FE9981', '#FE7864']
    )
    """
    midnights = generate_cmap(
        ['#000000',
         '#272367',
         '#3f5597',
         '#83afba',
         # '#d3e7f4',
         '#FFFFFF']
    )

    fearlessTV = generate_cmap(
        ['#171007',
         '#433016',
         '#95743b',
         '#c3a24f',
         '#ffffff']
    )

    cmap = midnights

    fontsize = 20
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    ax.set_title('$\\bf{'+str(JRM_NUM)+'}$'+'\n'+'Footprints', fontsize=fontsize, linespacing=0.85,
                 color='#3D4A7A', loc='left')
    ax.set_xlim(0, 360)
    ax.invert_xaxis()
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(['0', '90', '180', '270', '360'], fontsize=fontsize)
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_yticklabels(['-60', '-30', '0', '30', '60'], fontsize=fontsize)
    ax.set_xlabel('System III long. [deg]', fontsize=fontsize)
    ax.set_ylabel('Latitude [deg]', fontsize=fontsize)
    ax.tick_params(axis="x", which='major', direction='in')
    ax.tick_params(axis="x", which='minor', direction='inout')
    ax.tick_params(axis="y", which='major', direction='in')
    ax.tick_params(axis="y", which='minor', direction='inout')

    for i in satellite_index:
        latidx = 2*i+1
        s3idx = latidx+1

        ax.scatter(satellite_Nftp[:-1, s3idx],
                   satellite_Nftp[:-1, latidx],
                   label=satellite_name[i],
                   s=9,
                   marker=satellite_marker[i],
                   edgecolor=satellite_color[i],
                   facecolor='none',
                   zorder=10-i
                   )
        ax.scatter(satellite_Sftp[:-1, s3idx],
                   satellite_Sftp[:-1, latidx],
                   label='_nolegend_',
                   s=9,
                   marker=satellite_marker[i],
                   edgecolor=satellite_color[i],
                   facecolor='none',
                   zorder=10-i
                   )

    cs = ax.pcolormesh(wlong_arr, lat_arr,
                       # Babs_arr,
                       Dip_arr,
                       # cmap='YlGnBu_r',
                       cmap=cmap,
                       # vmin=0,
                       zorder=0.1
                       )
    cn = ax.contour(wlong_arr, lat_arr,
                    Dip_arr,
                    levels=3, colors='#ffffff')
    ax.clabel(cn)
    pp = fig.colorbar(cs)
    pp.ax.set_title(' ', fontsize=fontsize)
    pp.set_label('Dip angle [deg]', fontsize=fontsize)
    pp.ax.tick_params(labelsize=fontsize)
    ax.legend(markerscale=2, loc='upper right',
              bbox_to_anchor=(1, 1.14),
              fontsize=fontsize*0.5, ncol=4, frameon=False)
    fig.tight_layout()

    plt.savefig('img/JRM33/B_dip_angle_003.png')
    plt.show()

    return 0


if __name__ == '__main__':
    main()
