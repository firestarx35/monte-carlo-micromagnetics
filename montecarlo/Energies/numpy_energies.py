
import numpy as np


class Energies(object):


    def zeeman_energy(self):
        if self.zeeman_K is None:
            return 0
        # create a grid shape validation for length of grids shape less than 2 and more than 4 are not supported
        if len(self.grid.shape) < 2 or len(self.grid.shape) > 4:
            raise ValueError("Grid shape is not supported")

        energy = -self.m0*np.sum(np.dot(self.grid, self.zeeman_K)*self.dx*self.dy*self.dz)
        return energy
    
    def anisotropic_energy(self): # only uniaxial anisotropy is supported
        if self.anisotropic_K is None:
            return 0
        
        energy = np.sum(self.grid * self.anisotropic_u[::-1], axis=-1) # Dot product of m and u vector (#TODO: Why did I revese the u vector?)
        energy = energy**2 # Square of dot product of m and u vector
        energy = self.anisotropic_K*np.sum(energy)*self.dx*self.dy*self.dz #total energy of the system

        return energy

    def exchange_energy(self):
        if self.exchange_A is None:
            return 0
        
        # m = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
        # laplacian_m = scipy.ndimage.laplace(m)

        # compute the Laplacian
        laplacian_M = np.empty_like(self.grid,dtype='float64') 

        for i in range(3): #calculate laplacian for each component of m vector with respect to each axis
            laplacian_M[..., i] = (np.gradient(np.gradient(self.grid[..., i], self.dx, axis=0), self.dx, axis=0) #second order gradient with respect to x axis
                                + np.gradient(np.gradient(self.grid[..., i], self.dy, axis=1), self.dy, axis=1) #second order gradient with respect to y axis
                                + np.gradient(np.gradient(self.grid[..., i], self.dz, axis=2), self.dz, axis=2) #second order gradient with respect to z axis
                                )
        
        energy = np.sum(self.grid*laplacian_M, axis=-1) #dot product of m and laplacian_M
        energy = -self.exchange_A*np.sum(energy)*self.dx*self.dy*self.dz #total energy of the system

        return energy
            
    def dmi_energy(self):
        if self.dmi_D is None:
            return 0

        if self.type == 'Cnv_z':
            #calculate gradient of z component of m vector with respect to each axis
            gradM_z = np.empty_like(self.grid, dtype='float64')
            gradM_z[...,0] = np.gradient(self.grid[...,2],self.dx, axis=0) #dmz/dx
            gradM_z[...,1] = np.gradient(self.grid[...,2],self.dy, axis=1) #dmz/dy
            gradM_z[...,2] = np.gradient(self.grid[...,2],self.dz, axis=2) #dmz/dz

            #divergence of m vector
            div_M = np.empty_like(self.grid, dtype='float64')
            div_M[...,0] = np.gradient(self.grid[...,0],self.dx, axis=0) #dmx/dx
            div_M[...,1] = np.gradient(self.grid[...,1],self.dy, axis=1) #dmy/dy
            div_M[...,2] = np.gradient(self.grid[...,2],self.dz, axis=2) #dmz/dz
            div_M = np.sum(div_M, axis=-1) #Divergence of m vector
            
            m_del_mz = np.sum(self.grid*gradM_z, axis=-1) #dot product of m and gradient of z component of m vector
            mz_div_m = self.grid[...,2]*div_M  # mz∇⋅m

            energy = self.dmi_D * (np.sum(m_del_mz) - np.sum(mz_div_m)) * self.dx * self.dy * self.dz #Total energy of the system
        
        elif self.type == 'D2d_z':
            #not implemented yet
            energy = 0


        else:    
            curl = np.empty_like(self.grid)
            curl[..., 0] = np.gradient(self.grid[..., 2], self.dy, axis=1) - np.gradient(self.grid[..., 1], self.dz, axis=2)
            curl[..., 1] = np.gradient(self.grid[..., 0], self.dz, axis=2) - np.gradient(self.grid[..., 2], self.dx, axis=0)
            curl[..., 2] = np.gradient(self.grid[..., 1], self.dx, axis=0) - np.gradient(self.grid[..., 0], self.dy, axis=1)
            energy= np.sum(self.grid*curl, axis=-1) # dot product of m and curl
            energy = self.dmi_D*np.sum(energy)*self.dx*self.dy*self.dz # total energy of the system

        return energy
    
    def total_energy(self):
        return self.zeeman_energy() + self.anisotropic_energy() + self.exchange_energy() + self.dmi_energy()
