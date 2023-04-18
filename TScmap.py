""" TScmap.py

Created on Mar 15, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Mar 15, 2023)

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする


class TScmap():
    def __init__(self):
        return None

    def generate_cmap(self, colors):
        """自分で定義したカラーマップを返す"""
        values = range(len(colors))

        vmax = np.ceil(np.max(values))
        color_list = []
        for v, c in zip(values, colors):
            color_list.append((v / vmax, c))

        cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list)
        return cmap

    def MIDNIGHTS(self):
        midnights = self.generate_cmap(
            ['#000000',
             '#272367',
             '#3f5597',
             '#83afba',
             # '#d3e7f4',
             '#FFFFFF']
        )
        return midnights

    def FEARLESSTV(self):
        fearlessTV = self.generate_cmap(
            ['#171007',
             '#433016',
             '#95743b',
             '#c3a24f',
             '#ffffff']
        )
        return fearlessTV
