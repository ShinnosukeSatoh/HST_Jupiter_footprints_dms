""" B_JRM33.py

Created on Mar 18, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Apr 7, 2023)

"""


# %% LIBRARIES
import numpy as np
import pyshtools as pysh


# TOGGLES
JRM_NUM = 'JRM33'                           # 'JRM09' or 'JRM33'
SATELLITE = ['AM', 'IO', 'EU', 'GA']              # 'AM', 'IO', 'EU', 'GA'

# DATA LOADING FOR COEFFICIENTS AND POSITION OF SATELLITE FOOTPRINTS
jrm_coef = np.loadtxt(
    'data/'+JRM_NUM+'/coef.txt', skiprows=1, usecols=1
)

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


class B():
    def __init__(self):
        return None

    def JRM33(self, rs, theta, phi):
        """
        ### Parameters
        `rs` ... <float> radial distance [m] \\
        `theta` ... <float> colatitude of the point [deg] \\
        `phi` ... <float> eest-longitude of the point [deg] \\

        ### Returns
        <ndarray, shape (3,)> Magnetic field (B_r, B_theta, B_phi) [G]
        """

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
