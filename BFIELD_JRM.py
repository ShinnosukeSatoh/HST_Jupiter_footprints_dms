""" BFIELD_JRM.py

Created on Mar 18, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Mar 18, 2023)
2.0.0 (Mar 20, 2023) JRM33にも対応。衛星のfootprint位置も表示。

"""

# %% LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
import pyshtools as pysh
import math
import time


# %% matplotlib フォント設定
plt.rcParams.update({'font.sans-serif': "Arial",
                     'font.family': "sans-serif",
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:italic:bold'
                     })


# %% TOGGLES
JRM_NUM = 'JRM33'                           # 'JRM09' or 'JRM33'
SATELLITE = ['AM', 'IO', 'EU']              # 'AM', 'IO', 'EU', 'GA'


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
satellite_color = ['#888888', '#000000', '#FE0000', '#888888']
satellite_marker = ['o', '*', 's', 'd']
satellite_index = [satellite_name.index(i) for i in (SATELLITE)]

# DEGREE OF LEGENDRE FUNCTIONS
if JRM_NUM == 'JRM09':
    NN = 10
elif JRM_NUM == 'JRM33':
    NN = 30


# %% CONSTANTS
RJ = 71492E+3           # JUPITER RADIUS [m]

mu0 = 1.26E-6           # PERMEABILITY [N A^-2] = [kg m s^-2 A^-2]

me = 9.1E-31            # MASS OF ELECTRON [kg]
e = (1.6E-19)           # CHARGE OF ELECTRON [C]


# %% MAIN FUNCTION
def main():
    # LOCATION ON THE SURFACE
    """
    lat = 60.97                # LATITUDE [deg]
    wlong = 212.32              # WEST LONGITUDE [deg]

    B_JRM09(lat, wlong)
    """

    # SATELLITE SYS3 LONG
    # sate_s3 = satellite_Nftp[:, 0]

    # B FIELD MAPPING
    lat_arr = np.linspace(-89.85, 89.85, 90)
    wlong_arr = np.linspace(0, 360, lat_arr.size*2)
    x, y = np.meshgrid(wlong_arr, lat_arr)

    B_abs_arr = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            wlong = x[i, j]
            lat = y[i, j]
            Bvec = B_JRM(lat, wlong)*1E-5
            B_abs_arr[i, j] = np.sqrt(Bvec[0]**2 + Bvec[1]**2 + Bvec[2]**2)

    # PLOT
    """
    cmapBR = generate_cmap(
        ['#6692F7', '#87ADF9', '#A5C9FA', '#DAFEFE', '#FFFEE3', '#FAE3CB', '#F6C7B0', '#F2AA94', '#EF8C76'])
    """
    fontsize = 18
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    ax.set_title(JRM_NUM, fontsize=fontsize, weight='bold', loc='left')
    ax.set_xlim(0, 360)
    ax.invert_xaxis()
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(['0', '90', '180', '270', '360'], fontsize=fontsize)
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_yticklabels(['-60', '-30', '0', '30', '60'], fontsize=fontsize)
    ax.set_xlabel('System III long. [deg]', fontsize=fontsize)
    ax.set_ylabel('Latitude [deg]', fontsize=fontsize)

    for i in satellite_index:
        latidx = 2*i+1
        s3idx = latidx+1
        ax.scatter(satellite_Nftp[:, s3idx],
                   satellite_Nftp[:, latidx],
                   label=satellite_name[i],
                   s=9,
                   marker=satellite_marker[i],
                   edgecolor=satellite_color[i],
                   facecolor='none',
                   zorder=10-i
                   )
        ax.scatter(satellite_Sftp[:, s3idx],
                   satellite_Sftp[:, latidx],
                   label='_nolegend_',
                   s=9,
                   marker=satellite_marker[i],
                   edgecolor=satellite_color[i],
                   facecolor='none',
                   zorder=10-i
                   )

    cs = ax.pcolormesh(wlong_arr, lat_arr, B_abs_arr,
                       cmap='YlGnBu_r', zorder=0.1
                       )
    pp = fig.colorbar(cs)
    pp.ax.set_title(' ', fontsize=fontsize)
    pp.set_label('Intensity [G]', fontsize=fontsize)
    pp.ax.tick_params(labelsize=fontsize)
    ax.legend(markerscale=2, loc='upper right',
              bbox_to_anchor=(1, 1.12),
              fontsize=fontsize*0.5, ncol=4, frameon=False)
    fig.tight_layout()

    plt.savefig('img/'+JRM_NUM+'/B_surface_'+JRM_NUM+'_002.jpg')
    plt.show()

    return 0


# %% OTHER FUNCTIONS
def B_JRM(lat, wlong):
    """
    ### Parameters
    `lat` ... <float> Latitude of the point [deg] \\
    `wlong` ... <float> West-longitude of the point [deg] \\

    ### Returns
    <ndarray, shape (3,)> Magnetic field (B_r, B_theta, B_phi)
    """

    theta = np.radians(90-lat)      # [rad]
    phi = np.radians(-wlong)        # [rad]

    # RADIUS OF SURFACE (1/15 DYNAMICALLY FLATTENED SURFACE)
    rs = RJ*np.sqrt(np.cos(np.radians(lat))**2 +
                    (np.sin(np.radians(lat))*14.4/15.4)**2)

    # SCHMIDT QUASI-NORMALIZED LEGENDRE FUNCTIONS
    p_arr, dp_arr = pysh.legendre.PlmSchmidt_d1(NN, np.cos(theta))
    dp_arr *= -np.sin(theta)        # NECESSARY MULTIPLICATION

    p_arr = p_arr[1:]               # n <= 1
    dp_arr = dp_arr[1:]             # n <= 1

    # 磁場の計算
    # r成分
    dVdr = 0
    dVdr_n = np.zeros(NN)

    # theta成分
    dVdtheta = 0
    dVdtheta_n = np.zeros(NN)

    # phi成分
    dVdphi = 0
    dVdphi_n = np.zeros(NN)

    dVdr = 0
    dVdr_n = np.zeros(NN)

    for i in range(NN):
        n = i+1                              # INDEX n
        m = np.arange(0, n+1, 1, dtype=int)  # INDEX m

        p_s = int((n-1)*(n+2)/2)             # LEGENDRE関数arrayの先頭
        p_e = p_s + n                        # LEGENDRE関数arrayの終端
        g_s = n**2 - 1                       # g_nm の先頭
        g_e = g_s + n                        # g_nm の終端
        h_s = g_e + 1                        # g_nm の先頭
        h_e = h_s + (n-1)                    # g_nm の終端
        # print(n, m, g_s+1, g_e+1, h_s+1, h_e+1)

        P_nm = p_arr[p_s:p_e+1]
        dP_nm = dp_arr[p_s:p_e+1]
        g_nm = jrm_coef[g_s:g_e+1]
        h_nm = np.zeros(g_nm.shape)          # m = 0のゼロを作る
        h_nm[1:] = jrm_coef[h_s:h_e+1]       # m >= 1に値を格納する

        # INDEX m方向に和をとる
        dVdr_n[i] = (-1-n)*(RJ/rs)**(n+2) * np.sum(
            P_nm*(g_nm*np.cos(m*phi) + h_nm*np.sin(m*phi))
        )

        # INDEX m方向に和をとる
        dVdtheta_n[i] = (RJ/rs)**(n+2) * np.sum(
            dP_nm*(g_nm*np.cos(m*phi) + h_nm*np.sin(m*phi))
        )

        # INDEX m方向に和をとる
        dVdphi_n[i] = (1/np.sin(theta))*((RJ/rs)**(n+2)) * np.sum(
            P_nm*m*(-g_nm*np.sin(m*phi) + h_nm*np.cos(m*phi))
        )

        # print(P_nm.shape)
        # print(g_nm.shape)
        # print(h_nm.shape)
        # print(h_nm)

    # INDEX n方向に和をとる
    dVdr = -np.sum(dVdr_n)
    dVdtheta = -np.sum(dVdtheta_n)
    dVdphi = -np.sum(dVdphi_n)

    # print(dVdr*1E-5)
    # print(dVdtheta*1E-5)
    # print(dVdphi*1E-5)
    # print(np.sqrt(dVdr**2 + dVdtheta**2 + dVdphi**2)*1E-5)

    return np.array([dVdr, dVdtheta, dVdphi])


def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


# %% EXECUTE
if __name__ == '__main__':
    main()
