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

plt.figure(dpi = 200)
plt.plot(T_realK_MT, 1/M_emu_MT, ".")
plt.show()