from numpy.core.function_base import linspace
import pandas as pd
from  scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import numpy as np
from matplotlib import pyplot as plt

MT_Tb = pd.read_csv("Tb_MT.dat", sep = "\t", decimal = ',')

T_realK_MT = MT_Tb["TrealK"] # TrealK - Temperatura próbki w K.
HmT_mi0_MT = MT_Tb["mi0HmT"] # mi0HmT - natężenie pola magnetycznego pomnożone przez μ0. μ0H (mT). 
sH_MT = MT_Tb["sH"] # sH - niepewność mi0HmT. 
M_emu_MT = MT_Tb["Memu"] # Memu - namagnesowanie w emu. 
sM_emu__MT = MT_Tb["sMemu"] # sMemu - niepewność namagnesowania.

data_new = MT_Tb["TrealK"][MT_Tb["TrealK"]>240]
print(data_new)
y_new = (HmT_mi0_MT/M_emu_MT)[195:]



def func(x, a, b):
    return a*x + b

popt2, pcov2 = curve_fit(func, data_new, y_new)

x_fit = np.linspace(240,293,1000)

y_fit = popt2[0]*x_fit + popt2[1]

plt.figure(dpi = 200)
plt.plot(data_new, y_new, ".", ms = 0.5)
plt.plot(x_fit,y_fit, "--", lw = 0.5)
plt.xlabel("Temperatura T[K]")
plt.grid()
plt.ylabel("Odwrotnośc podatności magnetycznej []")
plt.savefig("plots/podatnosc.png")

print(popt2[1])
print(pcov2[0][0])