import numpy as np


class Grid:
    def __init__(self, system):
        self.m0 = 4*np.pi*1e-7 # Tm/A
        
        magnitudes = np.linalg.norm(system.m.array, axis=-1)
        magnitudes[magnitudes == 0] = 1 #To avoid zero division error
        self.grid = system.m.array/magnitudes[..., np.newaxis] #normalise vectors to get m(r)
        
        self.dx, self.dy, self.dz = system.m.mesh.cell
        self.temperature = system.T
        # What about multiple similar energy terms????
        try:
            self.zeeman_K = system.energy.zeeman.H
        except:
            self.zeeman_K = None
        try:
            self.exchange_A =  system.energy.exchange.A
        except:
            self.exchange_A = None
        try:
            self.dmi_D = system.energy.dmi.D
            self.type = system.energy.dmi.crystalclass
        except:
            self.dmi_D = None
        try:
            self.anisotropic_K, self.anisotropic_u = (system.energy.uniaxialanisotropy.K, np.array(system.energy.uniaxialanisotropy.u))
        except:
            self.anisotropic_K, self.anisotropic_u = (None, None)
        try:
            self.demag_N = system.energy.demag.N
        except:
            self.demag_N = None
            