""" Leadangle_wave.py

Created on Jul 3, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Jul 3, 2023)

"""

# %%
from numba import jit
import numpy as np
import math
import copy
import B_JRM33 as BJRM
import B_equator as BEQ


# 定数
MU0 = 1.26E-6            # 真空中の透磁率
AMU = 1.66E-27           # [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
C = 2.99792E+8           # 光速 [m/s]
phiRH0 = math.radians(-65.8)      # [rad]     Connerney+2020
TILT0 = math.radians(6.7)         # [rad]


# %%
class Awave():
    def __init__(self) -> None:
        pass

    def tracetau(self, r_orbit, S3wlong0, rho0, Hp, NS: str):

        if NS == 'N':
            ns = -1
        else:
            ns = 1

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_orbit)                # [m]
        theta = np.pi/2             # [rad]
        phi = 2*np.pi-S3wlong0      # [rad]
        x = rs*math.sin(theta)*math.cos(phi)
        y = rs*math.sin(theta)*math.sin(phi)
        z = rs*math.cos(theta)

        # 磁軸に合わせる
        y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
        z0 = z

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転
        zS3RH0 = y0*math.sin(TILT0) + z0*math.cos(TILT0)
        print(math.degrees(S3wlong0), zS3RH0/Hp)

        # 伝搬時間
        tau = 0     # [sec]

        dt = 1.0    # [sec]
        ds = 30000     # [m]
        for _ in range(8000000):
            Bv = BJRM.B().JRM33(rs, theta, phi)*1E-9        # [T]
            B0 = math.sqrt(Bv[0]**2+Bv[1]**2+Bv[2]**2)      # [T]
            Bv *= (B0-100E-9)/B0
            B0 += -100E-9

            Bx = Bv[0]*math.sin(theta)*math.cos(phi) \
                + Bv[1]*math.cos(theta)*math.cos(phi) \
                - Bv[2]*math.sin(phi)
            By = Bv[0]*math.sin(theta)*math.sin(phi) \
                + Bv[1]*math.cos(theta)*math.sin(phi) \
                + Bv[2]*math.cos(phi)
            Bz = Bv[0]*math.cos(theta) - Bv[1]*math.sin(theta)

            # 磁軸に合わせる
            y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
            z0 = z

            # S3RH で X3軸 の(右ネジ)まわりに-7度回転
            zS3RH = y0*math.sin(TILT0) + z0*math.cos(TILT0)

            # プラズマ質量密度 rho
            rho = rho0*AMU*(1E+6)*np.exp(-(zS3RH/Hp)**2)     # [m-3]

            # Alfven速度 Va
            Va = B0/math.sqrt(MU0*rho)    # [m/s]

            # 時間要素
            dt = ds/Va

            # 伝搬時間
            tau += dt   # [sec]

            dx = ds*Bx/B0
            dy = ds*By/B0
            dz = ds*Bz/B0

            x += dx*ns
            y += dy*ns
            z += dz*ns

            # 座標更新
            rs = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z/rs)
            phi = math.atan2(y, x)

            # L値
            # Lvalue = (rs/RJ)/((math.cos(0.5*np.pi-math.acos(zS3RH/rs)))**2)
            # Lvalue = zS3RH/Hp

            # print(math.degrees(S3wlong0), Lvalue, rho/(AMU*1E+6),
            #       math.degrees(math.acos(zS3RH/rs)), zS3RH/Hp)

            # print('      ', rs/RJ, zS3RH/RJ, Bx/B0, By/B0, Bz/B0)

            if Va/C > 0.03:
                print('       Too fast!', rs/RJ, rho/(AMU*1E+6))
                break

            if zS3RH > 1.1*Hp:
                print('      ', rs/RJ, rho/(AMU*1E+6), tau)
                break

        return tau

    def eqtau(self, r_orbit, S3wlong0, rho0, Hp, NS: str):

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_orbit)                # [m]
        theta = np.pi/2             # [rad]
        phi = 2*np.pi-S3wlong0      # [rad]
        x = rs*math.sin(theta)*math.cos(phi)
        y = rs*math.sin(theta)*math.sin(phi)
        z = rs*math.cos(theta)

        # 磁軸に合わせる
        y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
        z0 = z

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転
        zS3RH0 = y0*math.sin(TILT0) + z0*math.cos(TILT0)

        # トレースの向き
        if zS3RH0 < 0:
            ns = -1     # 北向き
        else:
            ns = 1      # 南向き

        # 伝搬時間
        tau = 0     # [sec]

        # 刻み
        dt = 1.0    # [sec]
        ds = 10000     # [m]
        for _ in range(8000000):
            Bv = BJRM.B().JRM33(rs, theta, phi)*1E-9        # [T]
            B0 = math.sqrt(Bv[0]**2+Bv[1]**2+Bv[2]**2)      # [T]

            Bx = Bv[0]*math.sin(theta)*math.cos(phi) + Bv[1] * \
                math.cos(theta)*math.cos(phi) - Bv[2]*math.sin(phi)
            By = Bv[0]*math.sin(theta)*math.sin(phi) + Bv[1] * \
                math.cos(theta)*math.sin(phi) + Bv[2]*math.cos(phi)
            Bz = Bv[0]*math.cos(theta) - Bv[1]*math.sin(theta)

            # 磁軸に合わせる
            y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
            z0 = z

            # S3RH で X3軸 の(右ネジ)まわりに-7度回転 (Centrifugal座標系に)
            zS3RH = y0*math.sin(TILT0) + z0*math.cos(TILT0)

            # プラズマ質量密度 rho
            rho = rho0*AMU*(1E+6)*np.exp(-(zS3RH/Hp)**2)     # [m-3]

            # Alfven速度 Va
            Va = B0/math.sqrt(MU0*rho)    # [m/s]
            Va = 500E+3                   # [m/s]

            # 時間要素
            dt = ds/Va

            # 伝搬時間
            tau += dt   # [sec]

            dx = Va*dt*Bx/B0
            dy = Va*dt*By/B0
            dz = Va*dt*Bz/B0

            x += dx*ns
            y += dy*ns
            z += dz*ns

            # 座標更新
            rs = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z/rs)
            phi = math.atan2(y, x)

            # L値
            # Lvalue = (rs/RJ)/((math.cos(0.5*np.pi-math.acos(zS3RH/rs)))**2)
            Lvalue = zS3RH/Hp

            # print(math.degrees(S3wlong0), Lvalue, rho/(AMU*1E+6),
            #       math.degrees(math.acos(zS3RH/rs)), zS3RH/Hp)

            if zS3RH*ns < 0:
                h = 2.5*Hp - zS3RH0
                tau = h/Va
                print(rs/RJ, rho/(AMU*1E+6), zS3RH, Va/C, tau)
                break

        return tau

    def tracefield(self, r_orbit: float, S3wlong0: float):
        """
        `r_orbit` Europa公転距離 [m] \\
        `S3wlong0` EuropaのSystem III経度 [rad] \\
        Europaの位置から遠心力赤道まで磁力線をトレースし、沿磁力線の距離S0を計算する。
        """

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_orbit)                # [m]
        theta = np.pi/2             # [rad]
        phi = 2*np.pi-S3wlong0      # [rad]
        x = rs*math.sin(theta)*math.cos(phi)
        y = rs*math.sin(theta)*math.sin(phi)
        z = rs*math.cos(theta)

        # 磁軸に合わせる
        y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
        z0 = z

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転
        zS3RH0 = y0*math.sin(TILT0) + z0*math.cos(TILT0)

        # 伝搬時間
        tau = 0     # [sec]

        # 線要素
        ds = 50000     # [m]

        # Europaの磁力線座標
        S0 = 0.

        # 遠心力赤道までトレースして磁力線に沿った現在地座標を調べる
        if zS3RH0 < 0:
            lineNS = -1     # 北向きにトレースする
        else:
            lineNS = 1      # 南向きにトレースする
        for _ in range(800000):
            Bv = BJRM.B().JRM33(rs, theta, phi)*1E-9        # [T]
            Bx = Bv[0]*math.sin(theta)*math.cos(phi) \
                + Bv[1]*math.cos(theta)*math.cos(phi) \
                - Bv[2]*math.sin(phi)
            By = Bv[0]*math.sin(theta)*math.sin(phi) \
                + Bv[1]*math.cos(theta)*math.sin(phi) \
                + Bv[2]*math.cos(phi)
            Bz = Bv[0]*math.cos(theta) - Bv[1]*math.sin(theta)

            Bcs = BJRM.B().BCS(x, y, z, phi)  # [nT]
            Bx += Bcs[0]*1E-9
            By += Bcs[1]*1E-9
            Bz += Bcs[2]*1E-9

            B0 = math.sqrt(Bx**2+By**2+Bz**2)      # [T]

            # 磁軸に合わせる
            y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
            z0 = z

            # S3RH で X3軸 の(右ネジ)まわりに-7度回転
            zS3RH = y0*math.sin(TILT0) + z0*math.cos(TILT0)

            # 座標更新 (x, y, z)
            x += ds*Bx/B0*lineNS
            y += ds*By/B0*lineNS
            z += ds*Bz/B0*lineNS

            # 座標更新 (r, theta, phi)
            rs = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z/rs)
            phi = math.atan2(y, x)

            # 座標更新 (沿磁力線: S0)
            S0 += ds*lineNS

            # print(zS3RH/RJ)

            if abs(zS3RH) < ds:
                # print(rs/RJ, S0/RJ)
                break

        return S0

    def tracefield2(self, r_orbit: float, S3wlong0: float, S0, rho0: float, Hp: float, NS: str):
        """
        `r_orbit` Europa公転距離 [m] \\
        `S3wlong0` EuropaのSystem III経度 [rad] \\
        Europaの位置から遠心力赤道まで磁力線をトレースし、沿磁力線の距離S0を計算する。
        """
        Niter = int(20000)

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_orbit)                # [m]
        theta = np.pi/2             # [rad]
        phi = 2*np.pi-S3wlong0      # [rad]
        x = rs*math.sin(theta)*math.cos(phi)
        y = rs*math.sin(theta)*math.sin(phi)
        z = rs*math.cos(theta)

        # 磁軸に合わせる
        y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
        z0 = z

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転
        zS3RH0 = y0*math.sin(TILT0) + z0*math.cos(TILT0)
        print(math.degrees(S3wlong0), zS3RH0/Hp, S0/Hp)

        # 伝搬時間
        tau = 0     # [sec]

        # 線要素
        ds = 50000     # [m]

        # Alfven速度を格納
        Va_arr = np.zeros(Niter)

        # Moon latitude (theta)を格納
        theta_arr = np.zeros(Niter)

        # 電離圏の方角
        if NS == 'N':
            ns = -1     # 北向き
        else:
            ns = 1      # 南向き
        for i in range(Niter):
            Bv = BJRM.B().JRM33(rs, theta, phi)*1E-9        # [T]
            Bx = Bv[0]*math.sin(theta)*math.cos(phi) \
                + Bv[1]*math.cos(theta)*math.cos(phi) \
                - Bv[2]*math.sin(phi)
            By = Bv[0]*math.sin(theta)*math.sin(phi) \
                + Bv[1]*math.cos(theta)*math.sin(phi) \
                + Bv[2]*math.cos(phi)
            Bz = Bv[0]*math.cos(theta) - Bv[1]*math.sin(theta)

            Bcs = BJRM.B().BCS(x, y, z, phi)  # [nT]
            Bx += Bcs[0]*1E-9
            By += Bcs[1]*1E-9
            Bz += Bcs[2]*1E-9

            B0 = math.sqrt(Bx**2+By**2+Bz**2)      # [T]

            # 磁軸に合わせる
            y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
            z0 = z

            # S3RH で X3軸 の(右ネジ)まわりに-7度回転
            zS3RH = y0*math.sin(TILT0) + z0*math.cos(TILT0)

            # プラズマ質量密度 rho
            rho = rho0*AMU*(1E+6)*np.exp(-(S0/Hp)**2)     # [kg m-3]

            # Alfven速度 Va
            Va = B0/math.sqrt(MU0*rho)    # [m/s]

            # 時間要素
            dt = ds/Va

            # 伝搬時間
            tau += dt   # [sec]

            # 座標更新 (x, y, z)
            x += (ds*Bx/B0)*ns
            y += (ds*By/B0)*ns
            z += (ds*Bz/B0)*ns

            # 座標更新 (r, theta, phi)
            rs = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z/rs)
            phi = math.atan2(y, x)

            # 座標更新 (沿磁力線: S0)
            S0 += ds*(-ns)

            # 配列格納
            Va_arr[i] = Va          # [m/s]
            theta_arr[i] = theta    # [rad]

            # if Va/C > 0.2:
            #     print('       Too fast!', rs/RJ, rho/(AMU*1E+6))
            #     break

            # 電離圏の方角
            # if (NS == 'N') and (Va/C > 0.2):    # VAで基準
            if (NS == 'N') and (S0 > 2*Hp):
                print('      N', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, tau)
                break
            if (NS == 'S') and (S0 < 2*Hp):
                print('      S', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, tau)
                break

        # 値が格納されていない部分は削除
        Va_arr = Va_arr[:i]
        theta_arr = theta_arr[:i]

        return tau, theta_arr, Va_arr

# %%
