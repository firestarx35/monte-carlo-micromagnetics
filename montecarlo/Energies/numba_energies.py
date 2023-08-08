from numba import njit, prange
import numpy as np



@njit(fastmath=True, parallel=True)
def zeeman_energy(grid, zeeman_K, m0, Ms, dx, dy, dz):
    # create a grid shape validation for length of grids shape less than 2 and more than 4 are not supported
    energy = -m0 * Ms * np.sum(grid * zeeman_K) * dx * dy * dz
    return energy


@njit(fastmath=True, parallel=True)
def anisotropic_energy(grid, anisotropic_K, anisotropic_u, dx, dy, dz):
    # If anisotropic_K is None in your method, you might be checking if it is not defined or zero.
    # In this function, I will consider it as zero check, as we can't check for None in numba njit.
    if anisotropic_K == 0:
        return 0

    # calculate energy
    energy = np.cross(grid, anisotropic_u)
    energy = energy**2
    energy = anisotropic_K*np.sum(energy)*dx*dy*dz #total energy of the system

    return energy


@njit(fastmath=True, parallel=True)
def exchange_energy(grid, grid_pad, exchange_A, dx, dy, dz):
    if exchange_A == 0:
        return 0
    # grid_pad = np.pad(grid, ((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge') # 'constant'
    # calculate the laplacian using (f(x + h) + f(x - h) - 2f(x))h^2

    laplacian_M = (((grid_pad[2:, 1:-1, 1:-1] - 2 * grid_pad[1:-1, 1:-1, 1:-1] + grid_pad[:-2, 1:-1, 1:-1])/dx**2) + \
                    ((grid_pad[1:-1, 2:, 1:-1] - 2 * grid_pad[1:-1, 1:-1, 1:-1] + grid_pad[1:-1, :-2, 1:-1])/dy**2) + \
                    ((grid_pad[1:-1, 1:-1, 2:] - 2 * grid_pad[1:-1, 1:-1, 1:-1] + grid_pad[1:-1, 1:-1, :-2])/dz**2)
                )
                    
    energy = np.sum(grid*laplacian_M) #dot product of m and laplacian_M
    energy = -exchange_A*energy*dx*dy*dz #total energy of the system
    return energy


@njit(fastmath=True, parallel=True)
def exchange_energy2(grid, exchange_A, dx, dy,dz):
    if exchange_A == 0:
        return 0
    

# Separating by crytallographic class

@njit(fastmath=True, parallel=True)
def dmi_energy(grid, grid_pad, Mtype, dmi_D, dx, dy, dz):
    if dmi_D == 0:
        return 0
    # grid_pad = np.pad(grid, ((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge') # 'constant'

    # compute energy
    if Mtype == 'Cnv_z':
        gradM_z = np.empty_like(grid, dtype='float64')
        gradM_z[..., 0] = (grid_pad[2:, 1:-1, 1:-1, 2] - grid_pad[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
        gradM_z[..., 1] = (grid_pad[1:-1, 2:, 1:-1, 2] - grid_pad[1:-1, :-2, 1:-1, 2]) / (2 * dy)
        gradM_z[..., 2] = (grid_pad[1:-1, 1:-1, 2:, 2] - grid_pad[1:-1, 1:-1, :-2, 2]) / (2 * dz)

        #divergence of m vector
        div_M = (
            (grid_pad[2:, 1:-1, 1:-1, 0] - grid_pad[:-2, 1:-1, 1:-1, 0]) / (2 * dx) +
            (grid_pad[1:-1, 2:, 1:-1, 1] - grid_pad[1:-1, :-2, 1:-1, 1]) / (2 * dy) +
            (grid_pad[1:-1, 1:-1, 2:, 2] - grid_pad[1:-1, 1:-1, :-2, 2]) / (2 * dz)
            )
        
        m_del_mz = np.sum(grid*gradM_z, axis=-1) #dot product of m and gradient of z component of m vector
        mz_div_m = grid[..., 2]*div_M  # mz∇⋅m

        energy = dmi_D * (np.sum(m_del_mz) - np.sum(mz_div_m)) * dx * dy * dz #Total energy of the system
    
    return energy

    # elif type == 'Cnv_xy':

        #  if self.type == 'Cnv_z':
        #     #calculate gradient of z component of m vector with respect to each axis
        #     gradM_z = np.empty_like(self.grid, dtype='float64')
        #     gradM_z[...,0] = np.gradient(self.grid[...,2],self.dx, axis=0) #dmz/dx
        #     gradM_z[...,1] = np.gradient(self.grid[...,2],self.dy, axis=1) #dmz/dy
        #     gradM_z[...,2] = np.gradient(self.grid[...,2],self.dz, axis=2) #dmz/dz

        #     #divergence of m vector
        #     div_M = np.empty_like(self.grid, dtype='float64')
        #     div_M[...,0] = np.gradient(self.grid[...,0],self.dx, axis=0) #dmx/dx
        #     div_M[...,1] = np.gradient(self.grid[...,1],self.dy, axis=1) #dmy/dy
        #     div_M[...,2] = np.gradient(self.grid[...,2],self.dz, axis=2) #dmz/dz
        #     div_M = np.sum(div_M, axis=-1) #Divergence of m vector
            
        #     m_del_mz = np.sum(self.grid*gradM_z, axis=-1) #dot product of m and gradient of z component of m vector
        #     mz_div_m = self.grid[...,2]*div_M  # mz∇⋅m

        #     energy = self.dmi_D * (np.sum(m_del_mz) - np.sum(mz_div_m)) * self.dx * self.dy * self.dz #Total energy of the system


        
