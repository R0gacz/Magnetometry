
from numpy.core.function_base import linspace
import pandas as pd
from  scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt

MH_Ni = pd.read_csv("Ni_MH.csv", sep = "\t", decimal = ',')


#Ni o masie 0.0898 g i gęstości 8908 kg/m3.
m_Ni = 0.08898
rho_Ni = 8908

mu0 = 4 * np.pi * 10**(-7) #H/m - pezenikalnosc magn. prozni

HT_mu0 = MH_Ni["mi0HT"]  # miOHT - natężenie pola magnetycznego pomnożone przez μ0. μ0H (T). - indukcja magn.
sHT_mu0 = MH_Ni["SH"]  #SH - niepewność mi0H
M_emu = MH_Ni["Memu"]  #Memu - namagnesowanie w emu.
sM_emu = MH_Ni["SMemu"]  #SM - niepewność pomiaru SMemu
T_sampK = MH_Ni["TsampK"]  #TsamK - Temperatura w okolicy próbki w K. 
sT_samp = MH_Ni["STsamp"]  #STsam - niepewność Tsam.

# M(B[T]) [emu/cm3]
"""
plt.figure(dpi=100)
plt.plot(HT_mu0, M_emu, ".", markersize = 0.5)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja M[emu")
plt.show()
"""

#M(B) [A/m]

MH_Am = M_emu*rho_Ni*1000/m_Ni #[A/m]


plt.figure(dpi=100)
plt.plot(HT_mu0, MH_Am, ".", color = "black", markersize = 0.5)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.savefig("Ni_M(H)")
#plt.show()

# Interpolacja gornego fragmentu krzywej histerezy


minMH_index = np.min(np.where(MH_Am == np.min(MH_Am))).astype(np.int64)
maxMH_index = np.max(np.where(MH_Am == np.max(MH_Am))).astype(np.int64)
print( "Index of max:", maxMH_index, "Index of min:", minMH_index)

range_x =  np.flip(HT_mu0[maxMH_index:minMH_index])
range_y = np.flip(MH_Am[maxMH_index:minMH_index])

f = interp1d(range_x, range_y, kind = "linear")
print("Pozostałość magnetyczna", f(0))

x = np.linspace(HT_mu0[minMH_index-1],HT_mu0[maxMH_index+1], 1000)
print(x)
plt.figure(dpi=100)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5)
plt.plot(x,f(x), "--", lw = 0.5)
plt.plot(0,f(0) , "r+", ms = 3)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.savefig("Ni_M(H)_interp")
plt.show()



# Interpolacja gornego fragmentu krzywej histerezy


minMH_index = np.min(np.where(MH_Am == np.min(MH_Am))).astype(np.int64)
maxMH_index = np.max(np.where(MH_Am == np.max(MH_Am))).astype(np.int64)
print( "Index of max:", maxMH_index, "Index of min:", minMH_index)

range_x =  np.flip(HT_mu0[maxMH_index:minMH_index])
range_y = np.flip(MH_Am[maxMH_index:minMH_index])

f = interp1d(range_x, range_y, kind = "linear")
print("Pozostałość magnetyczna", f(0))

x = np.linspace(HT_mu0[minMH_index-1],HT_mu0[maxMH_index+1], 10000)


plt.figure(dpi=100)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5)
plt.plot(x,f(x), "--", lw = 0.5)
plt.plot(0,f(0) , "r+", ms = 3)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.savefig("Ni_M(H)_interp")

plt.figure(dpi=100)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5)
plt.plot(x,f(x), "--", lw = 0.5)
plt.plot(0,f(0) , "r+", ms = 3)
plt.axhline("--", color = "grey", lw = 0.5)
plt.axvline("--", color = "grey", lw = 0.5)
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.xlim([-0.01, 0.01])
plt.ylim([-10000000, 10000000])
plt.savefig("Ni_M(H)_interp")

plt.show()



#Aproxymation of up-part curve

def curv(x, Ms, a, b):

    y = Ms*((np.exp(a*x)+np.exp(-a*x))/(np.exp(a*x)-np.exp(-a*x))-1/(a*x))+b
    return y

popt, pcov = curve_fit(curv, range_x, range_y)

Ms_fited = popt[0]
a_fited = popt[1]
b_fited = popt[2]
print("Ms: ", Ms_fited, ", a: ", a_fited, ", b: ", b_fited)
print("Odchylenia std. dopasowanych parametrow:")
print(pcov)

fited_curve = curv(x, Ms_fited, a_fited, b_fited)

plt.figure(dpi=100)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5)
plt.plot(x,fited_curve, "--", lw = 0.5)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.savefig("Ni_M(H)_aprox")
plt.show()

"""
def curv(x, Ms, a, b):

    y = Ms*((np.exp(a*x)+np.exp(-a*x))/(np.exp(a*x)-np.exp(-a*x))-1/(a*x))+b
    return y

popt, pcov = curve_fit(curv, HT_mu0, M_emu)

Ms_fited = popt[0]
a_fited = popt[1]
b_fited = popt[2]
print("Ms: ", Ms_fited, ", a: ", a_fited, ", b: ", b_fited)
print("Odchylenia std. dopasowanych parametrow:")
print(pcov)

x = np.linspace(-25,25,1000)
fited_curve = curv(x, Ms_fited, a_fited, b_fited)

plt.figure(dpi=100)
plt.plot(x, fited_curve, ".")
plt.grid()
plt.show()

"""

#Aproxymation of low-part curve

def curv(x, a, b, c, d, s):

    y = s*x**4 + a*x**3 + b*x**2 + c*x + d
    return y

popt, pcov = curve_fit(curv, range_x[120:140], range_y[120:140])

a_fited = popt[0]
b_fited = popt[1]
c_fited = popt[2]
d_fited = popt[3]
s_fited = popt[4]
print(", a: ", a_fited, ", b: ", b_fited)
print("s(a): ",pcov[0][0], ", s(b): ", pcov[1][1])

fited_curve = curv(x, a_fited, b_fited, c_fited, d_fited, s_fited)

plt.figure(dpi=100)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5)
plt.plot(x,fited_curve, "--", lw = 0.5)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.savefig("Ni_M(H)")
plt.show()

plt.figure(dpi=100)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5)
plt.plot(x,fited_curve, "--", lw = 0.5)
plt.axhline("--", color = "grey", lw = 0.5)
plt.axvline("--", color = "grey", lw = 0.5)
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.xlim([-0.01, 0.01])
plt.ylim([-10000, 10000])
plt.savefig("Ni_M(H)_interp")