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
# import B_equator as BEQ


# 定数
MU0 = 1.26E-6            # 真空中の透磁率
AMU = 1.66E-27           # [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
OMGJ = 1.75868E-4        # JUPITER SPIN ANGULAR VELOCITY [rad/s]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]
phiRH0 = math.radians(-65.8)      # [rad]     Connerney+2020
TILT0 = math.radians(6.7)         # [rad]
Ai_H = 1.0               # 水素 [原子量]
Ai_O = 16.0              # 酸素 [原子量]
Ai_S = 32.0              # 硫黄 [原子量]


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

        # ダイポール座標系に持っていく
        # S3RH で Z3軸 の(右ネジ)まわりに-155.8度回転
        phiRH0 = math.radians(-155.8)    # Connerney+2020
        rvec0 = np.array([
            x*math.cos(phiRH0) - y*math.sin(phiRH0),
            x*math.sin(phiRH0) + y*math.cos(phiRH0),
            z
        ])

        # S3RH で Y3軸 の(右ネジ)まわりに-9度回転
        THETA_D = -TILT0
        rvec0 = np.array([
            rvec0[0]*math.cos(THETA_D) + rvec0[2]*math.sin(THETA_D),
            rvec0[1],
            -rvec0[0]*math.sin(THETA_D) + rvec0[2]*math.cos(THETA_D)
        ])

        zS3RH0 = rvec0[2]

        """# 古いバージョン
        # 磁軸に合わせる
        y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
        z0 = z

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転 (遠心力赤道)
        zS3RH0 = y0*math.sin(TILT0) + z0*math.cos(TILT0)"""

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
            Bx += Bcs[0]*1E-9       # [T]
            By += Bcs[1]*1E-9
            Bz += Bcs[2]*1E-9

            B0 = math.sqrt(Bx**2+By**2+Bz**2)      # [T]

            # ダイポール座標系に持っていく
            # S3RH で Z3軸 の(右ネジ)まわりに-155.8度回転
            rvec0 = np.array([
                x*math.cos(phiRH0) - y*math.sin(phiRH0),
                x*math.sin(phiRH0) + y*math.cos(phiRH0),
                z
            ])

            # S3RH で Y3軸 の(右ネジ)まわりに-9度回転
            rvec0 = np.array([
                rvec0[0]*math.cos(THETA_D) + rvec0[2]*math.sin(THETA_D),
                rvec0[1],
                -rvec0[0]*math.sin(THETA_D) + rvec0[2]*math.cos(THETA_D)
            ])

            zS3RH = rvec0[2]

            """# 古いバージョン
            # 磁軸に合わせる
            y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
            z0 = z

            # S3RH で X3軸 の(右ネジ)まわりに-7度回転
            zS3RH = y0*math.sin(TILT0) + z0*math.cos(TILT0)"""

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
        # y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
        # z0 = z

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転
        # zS3RH0 = y0*math.sin(TILT0) + z0*math.cos(TILT0)
        # print(math.degrees(S3wlong0), zS3RH0/Hp, S0/Hp)

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
            # y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
            # z0 = z

            # S3RH で X3軸 の(右ネジ)まわりに-7度回転
            # zS3RH = y0*math.sin(TILT0) + z0*math.cos(TILT0)

            # プラズマ質量密度 rho
            rho = rho0*AMU*(1E+6)*np.exp(-(S0/Hp)**2)     # [kg m-3]

            # Alfven速度 Va
            Va = B0/math.sqrt(MU0*rho)    # [m/s]

            # 相対論効果の考慮
            if Va/C > 0.07:
                Va = Va/math.sqrt(1+(Va/C)**2)

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
            if (NS == 'N') and (Va/C > 0.17):    # VAで基準
                # if (NS == 'N') and (S0 > 2.1*Hp):
                # print('      N', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, tau)
                break
            if (NS == 'S') and (Va/C > 0.17):    # VAで基準
                # print('      S', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, tau)
                break

        # 値が格納されていない部分は削除
        Va_arr = Va_arr[:i]
        theta_arr = theta_arr[:i]

        return tau, theta_arr, Va_arr

    def tracefield3(self,
                    r_orbit: float,
                    S3wlong0: float,
                    S0,
                    RHO0: float,
                    n0_SII: float, H_SII: float,
                    n0_SIII: float, H_SIII: float,
                    n0_SIV: float, H_SIV: float,
                    n0_OII: float, H_OII: float,
                    n0_OIII: float, H_OIII: float,
                    n0_HII: float, H_HII: float,
                    NS: str):
        """
        `r_orbit` Europa公転距離 [m] \\
        `S3wlong0` EuropaのSystem III経度 [rad] \\
        Europaの位置から遠心力赤道まで磁力線をトレースし、沿磁力線の距離S0を計算する。\\
        イオン種ごとにスケールハイトを変える。イオンの温度比は次で固定。\\
          S+, S++, S+++, O+, O++, H+ = 130:65:45:130:90:17\\
        電気的中性の条件から、数密度比も固定。\\
          S+, S++, S+++, O+, O++, H+ = 0.02:0.14:0.04:0.30:0.08:0.12\\
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
        # y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
        # z0 = z

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転
        # zS3RH0 = y0*math.sin(TILT0) + z0*math.cos(TILT0)
        # print(math.degrees(S3wlong0), zS3RH0/Hp, S0/Hp)

        # 伝搬時間
        t = 0     # [sec]

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

            Babs = math.sqrt(Bx**2+By**2+Bz**2)      # [T]

            # Alfven速度 Va
            Va = Va_calc(Babs, S0,
                         n0_SII, H_SII,
                         n0_SIII, H_SIII,
                         n0_SIV, H_SIV,
                         n0_OII, H_OII,
                         n0_OIII, H_OIII,
                         n0_HII, H_HII)

            # 時間要素
            dt = ds/Va

            # 伝搬時間
            t += dt   # [sec]

            # 座標更新 (x, y, z)
            x += (ds*Bx/Babs)*ns
            y += (ds*By/Babs)*ns
            z += (ds*Bz/Babs)*ns

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
            if (NS == 'N') and (Va/C > 0.17):    # VAで基準
                # if (NS == 'N') and (rho < 0.1*RHO0*((1E+6)*AMU)):
                # print('      N', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, Va/C)
                break
            if (NS == 'S') and (Va/C > 0.17):
                # print('      S', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, t)
                break

        # 値が格納されていない部分は削除
        Va_arr = Va_arr[:i]
        theta_arr = theta_arr[:i]

        return t, theta_arr, Va_arr

    def tracefield4(self,
                    r_orbit: float,
                    S3wlong0: float,
                    Ti_ave: float,
                    S0,
                    n0_SII: float, Ti_SII: float,
                    n0_SIII: float, Ti_SIII: float,
                    n0_SIV: float, Ti_SIV: float,
                    n0_OII: float, Ti_OII: float,
                    n0_OIII: float, Ti_OIII: float,
                    n0_HII: float, Ti_HII: float,
                    NS: str):
        """
        `r_orbit` Europa公転距離 [m] \\
        `S3wlong0` EuropaのSystem III経度 [rad] \\
        Europaの位置から遠心力赤道まで磁力線をトレースし、沿磁力線の距離S0を計算する。\\
        イオン種ごとにスケールハイトを変える。イオンの温度比は次で固定。\\
          S+, S++, S+++, O+, O++, H+ = 130:65:45:130:90:17\\
        電気的中性の条件から、数密度比も固定。\\
          S+, S++, S+++, O+, O++, H+ = 0.02:0.14:0.04:0.30:0.08:0.12\\
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
        # y0 = x*math.sin(phiRH0) + y*math.cos(phiRH0)
        # z0 = z

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転
        # zS3RH0 = y0*math.sin(TILT0) + z0*math.cos(TILT0)
        # print(math.degrees(S3wlong0), zS3RH0/Hp, S0/Hp)

        # 伝搬時間
        t = 0     # [sec]

        # 線要素
        ds = 50000     # [m]

        # Alfven速度を格納
        Va_arr = np.zeros(Niter)

        # Moon latitude (theta)を格納
        theta_arr = np.zeros(Niter)

        # 赤道面の磁場
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
        BEQ = math.sqrt(Bx**2+By**2+Bz**2)      # [T]

        # 電離圏の方角
        if NS == 'N':
            ns = -1     # 北向き
        else:
            ns = 1      # 南向き

        # トレース
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

            Babs = math.sqrt(Bx**2+By**2+Bz**2)      # [T]

            # 質量密度 [kg m-3]
            # イオン種: S+, S++, S+++, O+, O++, H+, S+(hot), O+(hot)
            Z_j = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 1.0])
            m_j = np.array([32.0, 32.0, 32.0, 16.0, 16.0, 1.0])*AMU
            n_j0 = np.array([n0_SII, n0_SIII, n0_SIV, n0_OII, n0_OIII, n0_HII])
            Ti_j = np.array([Ti_SII, Ti_SIII, Ti_SIV, Ti_OII, Ti_OIII, Ti_HII])
            rho = Rho_Mei1995(x, y, z, Babs, BEQ, Z_j, m_j, n_j0, Ti_j, Ti_ave)

            # Alfven速度 Va
            Va = Babs/math.sqrt(MU0*rho)    # [m/s]

            # 相対論効果の考慮
            if Va/C > 0.07:
                Va = Va/math.sqrt(1+(Va/C)**2)

            # 時間要素
            dt = ds/Va

            # 伝搬時間
            t += dt   # [sec]

            # 座標更新 (x, y, z)
            x += (ds*Bx/Babs)*ns
            y += (ds*By/Babs)*ns
            z += (ds*Bz/Babs)*ns

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
            if (NS == 'N') and (Va/C > 0.17):    # VAで基準
                # if (NS == 'N') and (rho < 0.1*RHO0*((1E+6)*AMU)):
                # print('      N', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, Va/C)
                break
            if (NS == 'S') and (Va/C > 0.17):
                # print('      S', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, t)
                break

        # 値が格納されていない部分は削除
        Va_arr = Va_arr[:i]
        theta_arr = theta_arr[:i]

        return t, theta_arr, Va_arr


# %%
@jit(nopython=True, fastmath=True)
def Va_calc(B0: float, S0: float,
            n0_SII: float, H_SII: float,
            n0_SIII: float, H_SIII: float,
            n0_SIV: float, H_SIV: float,
            n0_OII: float, H_OII: float,
            n0_OIII: float, H_OIII: float,
            n0_HII: float, H_HII: float
            ):

    # S+イオン数密度
    n_SII = n0_SII*np.exp(-(S0/H_SII)**2)           # [cm-3] 数密度
    n_SIII = n0_SIII*np.exp(-(S0/H_SIII)**2)        # [cm-3] 数密度
    n_SIV = n0_SIV*np.exp(-(S0/H_SIV)**2)           # [cm-3] 数密度
    n_OII = n0_OII*np.exp(-(S0/H_OII)**2)           # [cm-3] 数密度
    n_OIII = n0_OIII*np.exp(-(S0/H_OIII)**2)        # [cm-3] 数密度
    n_HII = n0_HII*np.exp(-(S0/H_HII)**2)           # [cm-3] 数密度

    # トータルの質量密度
    rho = (n_SII*Ai_S + n_SIII*Ai_S + n_SIV*Ai_S +
           n_OII*Ai_O + n_OIII*Ai_O + n_HII*Ai_H)*(1E+6)*AMU    # [kg m-3]

    # Alfven速度 Va
    Va = B0/math.sqrt(MU0*rho)    # [m/s]

    # 相対論効果の考慮
    if Va/C > 0.07:
        Va = Va/math.sqrt(1+(Va/C)**2)

    return Va


@jit(nopython=True, fastmath=True)
def Rho_Mei1995(x: float, y: float, z: float,
                B: float, BEQ: float,
                Z_j, m_j, n_j0, Ti_j, Ti_ave: float,
                r0=9.4*RJ, R0=9.4*RJ
                ):
    """
    イオン種: S+, S++, S+++, O+, O++, H+, (S+(hot), O+(hot))
    """

    # 座標
    R = math.sqrt(x**2+y**2)
    r = math.sqrt(R**2+z**2)

    # イオン6種の価数リスト Z_j [価数]

    # イオン6種の質量リスト m_j [kg]

    # イオン6種の赤道密度リスト [cm-3]

    # CRITICAL PARAMETER TAU
    Zi_ave = 18                         # [原子量]
    Te_para = 20/2                      # 電子温度 (parallel) [eV]
    Ti_ave_para = Ti_ave/2              # イオン平均温度 (parallel) [eV]
    tau = Zi_ave*Te_para/Ti_ave_para    # 平均温度と平均密度で代用

    # イオン6種の温度リスト
    T_para = Ti_j/2
    T_perp = Ti_j/2

    # イオン6種のPhi_j配列
    Phi_j = (m_j/T_para)*G*MJ*(r0-r)/(r0*r) \
        + 0.5*(m_j/T_para)*(OMGJ**2)*(R**2 - R0**2) \
        + (1-(T_perp/T_para))*math.log(B/BEQ)

    # イオン6種の数密度配列 [cm-3]
    # n_j = n_j0*math.exp(Phi_j)*np.sum(Z_j*math.exp(Phi_j))**(-tau/(1+tau))

    # イオン6種の質量密度配列 [kg m-3]
    rho_j = n_j0*(1E+6)*m_j*np.exp(Phi_j) * \
        np.sum(Z_j*np.exp(Phi_j))**(-tau/(1+tau))

    # トータルのイオン質量密度 [kg m-3]
    rho = np.sum(rho_j)

    return rho
