import numpy as np


#Global constants
m0 = 4*np.pi*1e-7 # Tm/A


def zeeman_energy(grid, zeeman_K, Ms, dx, dy, dz):
    if zeeman_K is None:
        return 0
    if len(grid.shape) < 2 or len(grid.shape) > 4:
        raise ValueError("Grid shape is not supported")

    energy = -m0 * Ms * np.sum(np.dot(grid, zeeman_K) * dx * dy * dz)

    return energy


def anisotropic_energy(grid, anisotropic_K, anisotropic_u, dx, dy, dz):
    if anisotropic_K is None:
        return 0
    
    energy = np.cross(grid, anisotropic_u)
    energy = anisotropic_K * np.sum(energy**2) * dx * dy * dz

    return energy


def exchange_energy(grid, exchange_A, dx, dy, dz):
    if exchange_A is None:
        return 0
    pgrid = np.pad(grid, ((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge')

    laplacian_M = (((pgrid[2:, 1:-1, 1:-1] - 2 * pgrid[1:-1, 1:-1, 1:-1] + pgrid[:-2, 1:-1, 1:-1]) / dx**2) +
                   ((pgrid[1:-1, 2:, 1:-1] - 2 * pgrid[1:-1, 1:-1, 1:-1] + pgrid[1:-1, :-2, 1:-1]) / dy**2) +
                   ((pgrid[1:-1, 1:-1, 2:] - 2 * pgrid[1:-1, 1:-1, 1:-1] + pgrid[1:-1, 1:-1, :-2]) / dz**2))

    energy = np.sum(grid * laplacian_M)
    energy = -exchange_A * energy * dx * dy * dz

    return energy
        

def dmi_energy(grid, dmi_D, Dtype, dx, dy, dz):
    if dmi_D is None:
        return 0

    pgrid = np.pad(grid, ((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge')


    if Dtype == 'Cnv_z':
        #calculate gradient of z component of m vector with respect to each axis
        gradM_z = np.empty_like(grid, dtype='float64')
        gradM_z[..., 0] = (pgrid[2:, 1:-1, 1:-1, 2] - pgrid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
        gradM_z[..., 1] = (pgrid[1:-1, 2:, 1:-1, 2] - pgrid[1:-1, :-2, 1:-1, 2]) / (2 * dy)
        gradM_z[..., 2] = (pgrid[1:-1, 1:-1, 2:, 2] - pgrid[1:-1, 1:-1, :-2, 2]) / (2 * dz)

        #divergence of m vector
        div_M = ((pgrid[2:, 1:-1, 1:-1, 0] - pgrid[:-2, 1:-1, 1:-1, 0]) / (2 * dx) +
                (pgrid[1:-1, 2:, 1:-1, 1] - pgrid[1:-1, :-2, 1:-1, 1]) / (2 * dy) +
                (pgrid[1:-1, 1:-1, 2:, 2] - pgrid[1:-1, 1:-1, :-2, 2]) / (2 * dz)
            )
        
        m_del_mz = np.sum(grid*gradM_z, axis=-1) #dot product of m and gradient of z component of m vector
        mz_div_m = grid[..., 2]*div_M  # mz∇⋅m

        energy = m_del_mz - mz_div_m
    
    
    elif Dtype == 'D2d_z':

            gradM_x = np.empty_like(grid, dtype='float64')
            gradM_x[..., 0] = (pgrid[2:, 1:-1, 1:-1, 0] - pgrid[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
            gradM_x[..., 1] = (pgrid[1:-1, 2:, 1:-1, 1] - pgrid[1:-1, :-2, 1:-1, 1]) / (2 * dy)
            gradM_x[..., 2] = (pgrid[1:-1, 1:-1, 2:, 2] - pgrid[1:-1, 1:-1, :-2, 2]) / (2 * dz)
            
            gradM_y = np.empty_like(grid, dtype='float64')
            gradM_y[..., 0] = (pgrid[2:, 1:-1, 1:-1, 0] - pgrid[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
            gradM_y[..., 1] = (pgrid[1:-1, 2:, 1:-1, 1] - pgrid[1:-1, :-2, 1:-1, 1]) / (2 * dy)
            gradM_y[..., 2] = (pgrid[1:-1, 1:-1, 2:, 2] - pgrid[1:-1, 1:-1, :-2, 1]) / (2 * dz)


            #dm/dx x ^x - dm/dy x ^y
            cross_x = np.cross(gradM_x, np.array([1, 0, 0], dtype='float64'))
            cross_y = np.cross(gradM_y, np.array([0, 1, 0], dtype='float64'))
            res = cross_x - cross_y

            #Total energy
            energy =  grid*res
        
    elif type == 'Cnv_xy':
        # TODO: implement Cnv_xy
        raise NotImplementedError("Cnv_xy is not implemented yet")

    else:    
        curl = np.empty_like(grid, dtype='float64')
        curl[..., 0] = (pgrid[1:-1, 2:, 1:-1, 2] - pgrid[1:-1, :-2, 1:-1, 2]) / (2 * dy) - \
                                (pgrid[1:-1, 1:-1, 2:, 1] - pgrid[1:-1, 1:-1, :-2, 1]) / (2 * dz)
        
        curl[..., 1] = (pgrid[1:-1, 1:-1, 2:, 0] - pgrid[1:-1, 1:-1, :-2, 0]) / (2 * dz) - \
                                (pgrid[2:, 1:-1, 1:-1, 2] - pgrid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
        
        curl[..., 2] = (pgrid[2:, 1:-1, 1:-1, 1] - pgrid[:-2, 1:-1, 1:-1, 1]) / (2 * dx) - \
                                (pgrid[1:-1, 2:, 1:-1, 0] - pgrid[1:-1, :-2, 1:-1, 0]) / (2 * dy)
        
        energy= np.sum(grid*curl, axis=-1) # dot product of m and curl
    
    energy = np.sum(dmi_D[1:-1, 1:-1, 1:-1] * energy) * dx * dy * dz

    return energy


def numpy_total(grid, zeeman_K, exchange_A, anisotropic_K, anisotropic_u, dmi_D, Dtype, dx, dy, dz):
    return exchange_energy(grid, exchange_A, dx, dy, dz) + \
            anisotropic_energy(grid, anisotropic_K, anisotropic_u, dx, dy, dz) + \
            dmi_energy(grid, dmi_D, Dtype, dx, dy, dz) + \
            zeeman_energy(grid, zeeman_K, dx, dy, dz)