""" TEB_lead.py

Created on Oct 29, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Oct 29, 2023)

"""
# %%
from numba import jit
import numpy as np
import math
import copy
import B_JRM33 as BJRM


# 定数
qe = 1.6E-19        # 電荷 [C]
me = 9.1E-31        # 電子質量 [kg]
C = 2.99792E+8      # LIGHT SPEED [m/s]
A1 = qe/me
RJ = 71492*1E+3     # Jupiter radius [m]


# 常微分方程式
def ODE(RVvec, t):
    x = RVvec[0]
    y = RVvec[1]
    z = RVvec[2]
    R = math.sqrt(x**2+y**2)
    rs = math.sqrt(R**2+z**2)
    theta = math.acos(z/rs)
    phi = math.atan2(y, x)

    vx = RVvec[3]
    vy = RVvec[4]
    vz = RVvec[5]
    # print('vx, vy, vz = ', vx, vy, vz)

    Bv = BJRM.B().JRM33(rs, theta, phi)        # [nT]
    Bx = Bv[0]*math.sin(theta)*math.cos(phi) \
        + Bv[1]*math.cos(theta)*math.cos(phi) \
        - Bv[2]*math.sin(phi)
    By = Bv[0]*math.sin(theta)*math.sin(phi) \
        + Bv[1]*math.cos(theta)*math.sin(phi) \
        + Bv[2]*math.cos(phi)
    Bz = Bv[0]*math.cos(theta) - Bv[1]*math.sin(theta)

    Bcs = BJRM.B().BCS(x, y, z, phi)  # [nT]
    Bx += Bcs[0]
    By += Bcs[1]
    Bz += Bcs[2]

    Bx *= 1E-9
    By *= 1E-9
    Bz *= 1E-9

    B0 = math.sqrt(Bx**2+By**2+Bz**2)      # [T]

    return np.array([vx,
                     vy,
                     vz,
                     A1*(vy*Bz-vz*By),
                     A1*(vz*Bx-vx*Bz),
                     A1*(vx*By-vy*Bx)
                     ]), np.array([Bx, By, Bz]), B0


def RK4(V, S3latN, S3wlongN):
    """_summary_

    Args:
        S3latN (float): footprint latitude [rad]
        S3wlongN (float): footprint w longitude [rad]

    Returns:
        _type_: _description_
    """

    rs = 1.01*RJ                # 初期位置(北半球フットプリント)[m]
    theta = S3latN              # 初期位置(北半球フットプリント)[rad]
    phi = 2*np.pi-S3wlongN      # 初期位置(北半球フットプリント)[rad]

    x = rs*math.sin(theta)*math.cos(phi)
    y = rs*math.sin(theta)*math.sin(phi)
    z = rs*math.cos(theta)

    vx = -V*math.sin(math.radians(20))
    vy = V*math.cos(math.radians(20))
    vz = 0
    print('vx, vy, vz = ', vx, vy, vz)
    RVvec = np.array([x, y, z, vx, vy, vz])

    Bv = BJRM.B().JRM33(rs, theta, phi)        # [nT]
    Bx = Bv[0]*math.sin(theta)*math.cos(phi) \
        + Bv[1]*math.cos(theta)*math.cos(phi) \
        - Bv[2]*math.sin(phi)
    By = Bv[0]*math.sin(theta)*math.sin(phi) \
        + Bv[1]*math.cos(theta)*math.sin(phi) \
        + Bv[2]*math.cos(phi)
    Bz = Bv[0]*math.cos(theta) - Bv[1]*math.sin(theta)
    Bcs = BJRM.B().BCS(x, y, z, phi)  # [nT]
    Bx += Bcs[0]
    By += Bcs[1]
    Bz += Bcs[2]
    Bx *= 1E-9
    By *= 1E-9
    Bz *= 1E-9
    B0 = math.sqrt(Bx**2+By**2+Bz**2)      # [T]
    print('Bx, By, Bz = ', Bx*1E+9, By*1E+9, Bz*1E+9)

    # ピッチ角

    t = 0          # [sec]
    dt = 5E-11     # [sec]
    dt2 = dt*0.5   # [sec]

    NN = 50000000
    nn = 10000
    R_arr = np.zeros((int(NN/nn), 3))     # 結果を格納
    j = 0

    mu_B_arr = np.zeros(5)

    for i in range(NN):
        f1, Bvec, B0 = ODE(RVvec, t)
        f2, _, _ = ODE(RVvec+dt2*f1, t+dt2)
        f3, _, _ = ODE(RVvec+dt2*f2, t+dt2)
        f4, _, _ = ODE(RVvec+dt*f3, t+dt)
        RVvec += dt*(f1 + 2*f2 + 2*f3 + f4)/6
        t += dt

        x = RVvec[0]
        y = RVvec[1]
        z = RVvec[2]        # 南北
        r = math.sqrt(x**2 + y**2 + z**2)

        # ピッチ角
        vx, vy, vz = RVvec[3], RVvec[4], RVvec[5]
        v_para = (vx*Bvec[0] + vy*Bvec[1] + vz*Bvec[2])/B0
        v_perp = math.sqrt(vx**2+vy**2+vz**2 - v_para**2)
        pitch = math.atan2(v_perp, v_para)
        # print(v_perp/v_para)

        # 断熱不変量
        mu_B = me*(v_perp**2)/(2*B0)
        if i == 0:
            mu_B_arr[0] = mu_B

        # Gyro period
        TC = 2*np.pi*me/(qe*B0)

        if i % nn == 0:
            R_arr[j] = np.array([x, y, z])
            j += 1
            print(r/RJ, math.degrees(pitch), mu_B/mu_B_arr[0], t)

        dt = TC/100     # [sec]
        dt2 = dt*0.5    # [sec]

        # 北半球に到達したら終了
        if z > 0:
            if r < 1.0*RJ:
                print('North')
                break

        # 南半球に到達したら終了
        if z < 0:
            if r < 1.0*RJ:
                theta = math.acos(z/rs)
                phi = math.atan2(y, x)
                wlong = 2*np.pi-phi
                print('South', np.degrees(theta), np.degrees(wlong), t)
                break

    return R_arr


Te = 50000            # 電子エネルギー [eV]
V = math.sqrt((3*Te*1.602E-19)/me)      # 速度 [m/s]
RK4(V, math.radians(54.53), math.radians(180.69))
