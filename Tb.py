from numpy.core.function_base import linspace
import pandas as pd
from  scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import numpy as np
from matplotlib import pyplot as plt

MH_Tb = pd.read_csv("Tb_MH.csv", sep = "\t", decimal = ',')


#Tb o masie 0.0898 g i gęstości 8908 kg/m3 = 8.908 g/cm3.
m_Tb = 0.1177
rho_Tb = 8.219
M_A_Tb = 158.9253
N_A = 6.022*10**23
mu_b = 9.27401/10**(21)

mu0 = 4 * np.pi * 10**(-7) #H/m - pezenikalnosc magn. prozni

HT_mu0 = MH_Tb["mi0HT"]  # miOHT - natężenie pola magnetycznego pomnożone przez μ0. μ0H (T). - indukcja magn.
sHT_mu0 = MH_Tb["SH"]  #SH - niepewność mi0H
M_emu = MH_Tb["Memu"]  #Memu - namagnesowanie w emu.
sM_emu = MH_Tb["SMemu"]  #SM - niepewność pomiaru SMemu
T_sampK = MH_Tb["TsampK"]  #TsamK - Temperatura w okolicy próbki w K. 
sT_samp = MH_Tb["STsamp"]  #STsam - niepewność Tsam.

#M(Hmu0) [emu]
plt.figure(dpi=200)
plt.plot(HT_mu0, M_emu, ".",color = "black", markersize = 0.5)
plt.grid()
plt.xlabel("Natężenie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja M [emu]")
plt.savefig("plots/Tb_dane")


#M(Hmu0) [A/m]
MH_Am = M_emu*rho_Tb*1000/m_Tb #[A/m]
H = HT_mu0/mu0



plt.figure(dpi=200)
plt.plot(HT_mu0, MH_Am, ".", color = "black", markersize = 0.5)
plt.grid()
plt.xlabel("Natężenie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.savefig("plots/Tb_SI")



# Upper hysteresis interpolation

minHT_index_up = np.where(HT_mu0 == np.min(HT_mu0))[0]
maxHT_index_up = np.where(HT_mu0 == np.max(HT_mu0))[0]
range_index_up = np.arange(maxHT_index_up,minHT_index_up,1) # range of indexes for upper part of curve
print( "Range od index:", range_index_up[0], range_index_up[-1])

range_x_up =  np.flip(HT_mu0[range_index_up])
range_y_up = np.flip(MH_Am[range_index_up])

x_up = np.linspace(HT_mu0[minHT_index_up-1],HT_mu0[maxHT_index_up+1], 1000)

Mh_up = interp1d(range_x_up, range_y_up, kind = 'linear')
print("Pozostałość magnetyczna Tb", Mh_up(0))

Hm_up = interp1d(range_y_up, range_x_up,  kind = 'linear')
print("Pole koercji Tb: ", Hm_up(0))


plt.figure(dpi=200)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5, label = "Dane")
plt.plot(x_up,Mh_up(x_up), "--", lw = 0.5, label = "Interpolacja")
plt.plot(0,Mh_up(0) , "r+", ms = 6, label = "Remanencja B$_r$")
plt.plot(Hm_up(0)-0.0005, 0, "g+", ms = 6, label = "Pole koercji H$_c$")
plt.grid()
plt.legend()
plt.xlabel("Natężenie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.savefig("plots/Tb_interp")

plt.figure(dpi=200)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5, label = "Dane")
plt.plot(x_up,Mh_up(x_up), "--", lw = 0.5, label = "Interpolacja")
plt.plot(0,Mh_up(0) , "r+", ms = 6, label = "Remanencja B$_r$")
plt.plot(Hm_up(0)-0.0005, 0, "g+", ms = 6, label = "Pole koercji H$_c$")
plt.grid()
plt.legend()
plt.xlabel("Natężenie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.xlabel("Natężenie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.xlim([-0.03, 0.03])
plt.ylim([-30000, 30000])
plt.savefig("plots/Tb_max")

# Lower hysteresis interpolation

minHT_index_low = np.where(HT_mu0 == np.min(HT_mu0))[0]
maxHT_index_low = np.where(HT_mu0[minHT_index_low[0]:] == np.max(HT_mu0[minHT_index_low[0]:]))[0] + minHT_index_low

range_index_low = np.arange(minHT_index_low,maxHT_index_low,1)
print( "Range od index:", range_index_low[0], range_index_low[-1])

range_x_low =  HT_mu0[range_index_low]
range_y_low = MH_Am[range_index_low]

x_low = np.linspace(HT_mu0[minHT_index_low+1],HT_mu0[maxHT_index_low-1], 1000)

Mh_low = interp1d(range_x_low, range_y_low, kind = "linear")
print("Pozostałość magnetyczna", Mh_low(0))

Hm_low = interp1d(range_y_low, range_x_low,  kind = "linear")
print("Pole koercji: ", Hm_low(0))

#up and low
plt.figure(dpi=200)
plt.plot(HT_mu0, MH_Am, ".", color = "black", ms = 0.5, label = "Dane")
plt.plot(x_up,Mh_up(x_up), "b--", lw = 0.5, label = "Interpolacja")
plt.plot(0,Mh_up(0) , "r+", ms = 6, label = "Remanencja B$_r$")
plt.plot(Hm_up(0)-0.0005, 0, "g+", ms = 6, label = "Pole koercji H$_c$")
plt.plot(x_low,Mh_low(x_low), "b--", lw = 0.5)
plt.plot(0,Mh_low(0) , "r+", ms = 6)
plt.plot(Hm_low(0), 0, "g+", ms = 6)
plt.grid()
plt.legend()
plt.xlabel("Natężenie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.xlabel("Natężenie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.xlim([-0.03, 0.03])
plt.ylim([-30000, 30000])
plt.savefig("plots/Tb_max_and_min")


# Bc i Hr
sM_emu_Am = sM_emu*rho_Tb*1000/m_Tb #[A/m]

print("Pozostałość magnetyczna Terbu:", ((Mh_up(0)-Mh_low(0))/2),"$\pm$", np.mean(sM_emu_Am), "[A/m]")
print("Pole koercji Terbu: ", (-(Hm_up(0)-Hm_low(0))/2),"$\pm$", np.mean(sHT_mu0), "[T]")

#M(H) [mu_b/n]

M_bohr = (M_emu*M_A_Tb)/(m_Tb*N_A*mu_b)  


# Sum of upper and lower part

x_mean = np.linspace(HT_mu0[minHT_index_low+1], HT_mu0[maxHT_index_low-1],1000)

Mh_mean = (Mh_up(x_mean)+Mh_low(x_mean))/2

Mh_mean_bohr = (Mh_mean*M_A_Tb*m_Tb)/(m_Tb*N_A*mu_b*rho_Tb*1000)

x_mean = np.ndarray((x_mean.size), dtype=float, buffer=x_mean)
Mh_mean_bohr = np.ndarray((Mh_mean_bohr.size), dtype=float, buffer= Mh_mean_bohr)

plt.figure(dpi=200)
plt.plot(x_mean, Mh_mean_bohr, "b-", markersize = 1)
plt.plot(HT_mu0, M_bohr, ".", color = "black", markersize = 0.5)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [$\mu_{B}$/cząsteczkę]")
plt.savefig("plots/Tb_bohr")


# Aproximation 

def curv(x, Ms, a, b):

    y = Ms*((np.exp(a*x)+np.exp(-a*x))/(np.exp(a*x)-np.exp(-a*x))-1/(a*x))+b
    return y


popt, pcov = curve_fit(curv, x_mean, Mh_mean_bohr)

Ms_fited = popt[0]
a_fited = popt[1]
b_fited = popt[2]
print("Ms: ", Ms_fited, ", a: ", a_fited, ", b: ", b_fited)
print("s(Ms): ",pcov[0][0], ", s(a): ", pcov[1][1], ", s(b): ", pcov[2][2])
print("Namagnesowaniem nasycenia Ms Terbu: ", Ms_fited,"$\pm$",pcov[0][0])

x_fit = np.linspace(HT_mu0[minHT_index_low-1],HT_mu0[maxHT_index_low+1], 1000)
fited_curve = curv(x_fit, Ms_fited, a_fited, b_fited)

plt.figure(dpi=200)
plt.plot(x_mean, Mh_mean_bohr, ".", color = "black", ms = 0.3)
plt.plot(x_fit,fited_curve, "--", lw = 0.7)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [$\mu_{B}$/cząsteczkę]")
plt.savefig("plots/Tb_aprox")

# Enegry

Mh_diff = (Mh_up(x_mean)-Mh_low(x_mean))
E = np.sum(Mh_diff)

print(E)

plt.figure(dpi=200)
plt.plot(x_mean, Mh_diff, "b.", markersize = 1)
plt.grid()
plt.xlabel("Natężęie pola H${\mu_0}$ [T]")
plt.ylabel("Magnetyzacja próbki M [A/m]")
plt.savefig("plots/Tb_E")