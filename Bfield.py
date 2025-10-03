#%% Import necessary libraries and load data
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
from scipy.optimize import curve_fit
import scipy.special
from scipy.interpolate import RegularGridInterpolator
axesmat = sio.loadmat('AxesOut.mat')
XOut = axesmat['XOut'][:, 0]*1e3  # Convert to mm
YOut = axesmat['YOut'][:, 0]*1e3
ZOut = axesmat['ZOut'][:, 0]*1e3

bmat = sio.loadmat('BOut.mat')
BxOut = bmat['BxOut']
ByOut = bmat['ByOut']
BzOut = bmat['BzOut']
BOut = np.sqrt(BxOut**2 + ByOut**2 + BzOut**2)

X, Y, Z = np.meshgrid(XOut, YOut, ZOut, indexing='ij')
levels = np.arange(-40, 41, 2)

def helmholtz_field(x, y, z, I=1.0, coil_size=4e-3, offset=4e-3, n=3, mu0=4*np.pi*1e-7):
    """
    Calculate Bx, By, Bz at (x, y, z) for a pair of Helmholtz coils.
    - coil_size: radius of coils (meters)
    - offset: distance from origin to each coil along z (meters)
    - I: current through coils (Amps)
    - n: number of turns per coil
    Returns: Bx, By, Bz (same shape as x, y, z)
    """
    def single_loop_B(x, y, z, R, z0, I, n, mu0):
        # Field from a single loop centered at (0,0,z0)
        rho = np.sqrt(x**2 + y**2)
        zz = z - z0
        k_sq = 4*R*rho / ((R + rho)**2 + zz**2)
        k = np.sqrt(k_sq)
        # Avoid division by zero
        k = np.where(k == 0, 1e-12, k)

        K = np.vectorize(lambda m: scipy.special.ellipk(m))(k_sq)
        E = np.vectorize(lambda m: scipy.special.ellipe(m))(k_sq)
        coeff = mu0 * I * n / (2 * np.pi)
        Brho = coeff * zz / (rho * np.sqrt((R + rho)**2 + zz**2)) * (-K + (R**2 + rho**2 + zz**2)/((R - rho)**2 + zz**2)*E)
        Bz = coeff / np.sqrt((R + rho)**2 + zz**2) * (K + (R**2 - rho**2 - zz**2)/((R - rho)**2 + zz**2)*E)
        # Convert Brho to Bx, By
        with np.errstate(divide='ignore', invalid='ignore'):
            Bx = Brho * x / rho
            By = Brho * y / rho
            Bx = np.nan_to_num(Bx)
            By = np.nan_to_num(By)
        return Bx, By, Bz

    # Field from coil at +offset
    Bx1, By1, Bz1 = single_loop_B(x, y, z, coil_size, +offset, I, n, mu0)
    # Field from coil at -offset
    Bx2, By2, Bz2 = single_loop_B(x, y, z, coil_size, -offset, I, n, mu0)
    # Superpose
    Bx = Bx1 + Bx2
    By = By1 + By2
    Bz = Bz1 + Bz2
    return Bx, By, Bz

def helmholtz_fitting(coordinates, i, c, o):
    x, y, z = coordinates
    Bx, By, Bz = helmholtz_field(x, y, z, I=i, coil_size=c, offset=o, n=3)
    return np.sqrt(Bx**2 + By**2 + Bz**2)

def interpolation(x, y, z, X, Y, Z, B):
    Bx_interp = RegularGridInterpolator((X, Y, Z), B[0], bounds_error=False, fill_value=None)
    By_interp = RegularGridInterpolator((X, Y, Z), B[1], bounds_error=False, fill_value=None)
    Bz_interp = RegularGridInterpolator((X, Y, Z), B[2], bounds_error=False, fill_value=None)

    Xf, Yf, Zf = np.meshgrid(x, y, z, indexing='ij')

    Bx = Bx_interp((Xf, Yf, Zf))
    By = By_interp((Xf, Yf, Zf))
    Bz = Bz_interp((Xf, Yf, Zf))
    return Bx, By, Bz

#%% Fitting the simulated field to Helmholtz coil model
# Initial guess: I=5e4 A, coil_size=4e-3 m, offset=4e-3 m
popt, pcov = curve_fit(helmholtz_fitting, (X.ravel()*1e-3, Y.ravel()*1e-3, Z.ravel()*1e-3), BOut.ravel(), p0=[5e4, 4e-3, 4e-3], bounds=([1e4, 3e-3, 3e-3], [1e5, 5e-3, 5e-3]), maxfev=10000)
print(f"Fitted current: {popt[0]:.2f} A")
print(f"Fitted coil size: {popt[1]:.2f} m")
print(f"Fitted offset: {popt[2]:.2f} m")
Current = popt[0]
Coil_size = popt[1]
Offset = popt[2]

Bx, By, Bz = helmholtz_field(X.ravel()*1e-3, Y.ravel()*1e-3, Z.ravel()*1e-3, I=Current, coil_size=Coil_size, offset=Offset, n=3)
Bx = Bx.reshape(X.shape)
By = By.reshape(X.shape)
Bz = Bz.reshape(X.shape)

Z, Y = np.meshgrid(ZOut, YOut, indexing='ij')
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].contourf(Z, Y, BzOut[50, :, :], shading='auto', levels=levels, cmap='jet')
axs[0].set_title('MIFEDS Output')
axs[0].set_xlabel('Z [mm]')
axs[0].set_ylabel('Y [mm]')
fig.colorbar(axs[0].collections[0], ax=axs[0])
axs[1].contourf(Z, Y, Bz[50, :, :], shading='auto', levels=levels, cmap='jet')
axs[1].set_title('Simulated Helmholtz Coil')
axs[1].set_xlabel('Z [mm]')
axs[1].set_ylabel('Y [mm]')
fig.colorbar(axs[1].collections[0], ax=axs[1])
plt.suptitle('Comparison of Simulated and Calculated Bz Fields\nHelmholtz Coil Bz at X=0mm')
plt.show()

#%% Performing interpolation to FLASH grid
## For FLASH the z-axis is foil, and y-axis is along the coil axis
x_mifeds, y_mifeds, z_mifeds = XOut, ZOut, YOut
Bx_mifeds, By_mifeds, Bz_mifeds = np.transpose(BxOut, (0, 2, 1)), np.transpose(BzOut, (0, 2, 1)), np.transpose(ByOut, (0, 2, 1))

flash_step = 25*1e-3  # mm
x_flash = np.arange(-3.2-flash_step/2, 3.2+flash_step/2, flash_step)
y_flash = np.arange(-3.2-flash_step/2, 3.2+flash_step/2, flash_step)
z_flash = np.arange(-2.4-flash_step/2, 10.4+flash_step/2, flash_step) - 4  # Shift z to center of coil

Bx_flash, By_flash, Bz_flash = interpolation(x_flash, y_flash, z_flash, x_mifeds, y_mifeds, z_mifeds, [Bx_mifeds, By_mifeds, Bz_mifeds])

fig, axs = plt.subplots(3, 2, figsize=(12, 12))
Y_flash, Z_flash = np.meshgrid(y_flash, z_flash, indexing='ij')
Y_mifeds, Z_mifeds = np.meshgrid(y_mifeds, z_mifeds, indexing='ij')
axs[0, 0].contourf(Y_mifeds, Z_mifeds, Bx_mifeds[50, :, :], shading='auto', levels=levels, cmap='jet')
axs[0, 0].set_title('MIFEDS Output')
axs[0, 0].set_xlabel('Y [mm]')
axs[0, 0].set_ylabel('Z [mm]')
fig.colorbar(axs[0, 0].collections[0], ax=axs[0, 0], label='Bx [T]')
axs[0, 1].contourf(Y_flash, Z_flash, Bx_flash[50, :, :], shading='auto', levels=levels, cmap='jet')
axs[0, 1].set_title('Interpolated to FLASH Grid')
axs[0, 1].set_xlabel('Y [mm]')
axs[0, 1].set_ylabel('Z [mm]')
fig.colorbar(axs[0, 1].collections[0], ax=axs[0, 1], label='Bx [T]')

axs[1, 0].contourf(Y_mifeds, Z_mifeds, By_mifeds[50, :, :], shading='auto', levels=levels, cmap='jet')
axs[1, 0].set_xlabel('Y [mm]')
axs[1, 0].set_ylabel('Z [mm]')
fig.colorbar(axs[1, 0].collections[0], ax=axs[1, 0], label='By [T]')
axs[1, 1].contourf(Y_flash, Z_flash, By_flash[50, :, :], shading='auto', levels=levels, cmap='jet')
axs[1, 1].set_xlabel('Y [mm]')
axs[1, 1].set_ylabel('Z [mm]')
fig.colorbar(axs[1, 1].collections[0], ax=axs[1, 1], label='By [T]')

axs[2, 0].contourf(Y_mifeds, Z_mifeds, Bz_mifeds[50, :, :], shading='auto', levels=levels, cmap='jet')
axs[2, 0].set_xlabel('Y [mm]')
axs[2, 0].set_ylabel('Z [mm]')
fig.colorbar(axs[2, 0].collections[0], ax=axs[2, 0], label='Bz [T]')
axs[2, 1].contourf(Y_flash, Z_flash, Bz_flash[50, :, :], shading='auto', levels=levels, cmap='jet')
axs[2, 1].set_xlabel('Y [mm]')
axs[2, 1].set_ylabel('Z [mm]')
fig.colorbar(axs[2, 1].collections[0], ax=axs[2, 1], label='Bz [T]')

for ax in axs.flat:
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
plt.suptitle('Comparison of Simulated and Interpolated B Fields\nHelmholtz Coil B at X=0mm')
plt.show()


## Save to npz for FLASH input
X_flash, Y_flash, Z_flash = np.meshgrid(x_flash, y_flash, z_flash, indexing='ij')
np.savez('Bfield_flash.npz', x=X_flash, y=Y_flash, z=Z_flash, Bx=Bx_flash, By=By_flash, Bz=Bz_flash)

## Save to CSV for FLASH input <- These are horrendously large file, so npz is better (but leaving this here for reference)
# X, Y, Z = np.meshgrid(x_flash, y_flash, z_flash, indexing='ij')
# np.savetxt('coords.csv', np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), delimiter=',')
# np.savetxt('Bx.csv', Bx_flash.ravel(), delimiter=',')
# np.savetxt('By.csv', By_flash.ravel(), delimiter=',')
# np.savetxt('Bz.csv', Bz_flash.ravel(), delimiter=',')

#%% Reading back the npz file to verify
def read_npz(filename):
    data = np.load(filename)
    x = data['x']
    y = data['y']
    z = data['z']
    Bx = data['Bx']
    By = data['By']
    Bz = data['Bz']
    return x, y, z, Bx, By, Bz

x, y, z, Bx, By, Bz = read_npz('Bfield_flash.npz')
print(f"Loaded data from Bfield_flash.npz with shapes: x={x.shape}, y={y.shape}, z={z.shape}, Bx={Bx.shape}, By={By.shape}, Bz={Bz.shape}")
# %%
