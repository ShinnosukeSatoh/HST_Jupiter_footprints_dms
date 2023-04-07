import matplotlib
from matplotlib import font_manager
import matplotlib.font_manager as fm
for f in fm.fontManager.ttflist:
    print(f.name)
for f in fm.fontManager.afmlist:
    print(f.name)


print("List of all fonts currently available in the matplotlib:")
print(*font_manager.findSystemFonts(fontpaths=None, fontext='ttf'), sep="")

print(matplotlib.matplotlib_fname())
