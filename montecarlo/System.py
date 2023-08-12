import numpy as np
import micromagneticmodel as mm
from montecarlo.Driver.driver import driver_numba#, driver_numpy
import os
from pydantic import BaseModel, ValidationError




class Schedule(BaseModel):
    type: str = None
    start_temp: float = 0.01
    end_temp: float = 0.01
    steps: int = None


class System:
    def __init__(self, system: mm.System, Ms:float, schedule_name: str, schedule: dict):
        self.system = system
        self.schedule_name = schedule_name
        self.Ms = Ms
        magnitudes = np.linalg.norm(system.m.array, axis=-1)
        magnitudes[magnitudes == 0] = 1 #To avoid zero division error
        self.grid = system.m.array/magnitudes[..., np.newaxis] #normalise vectors to get m(r)
        self.dx, self.dy, self.dz = np.array(system.m.mesh.cell, dtype='float64') #cell dimensions
        try:
            self.schedule = Schedule(**schedule)
        except ValidationError as e:
            print(e)
        self.schedule_type = schedule['type']
        self.temperature = schedule['start_temp']
        self.end_temp = schedule['end_temp']
        self.steps = schedule['steps']
        self.dt = (self.end_temp - self.temperature)/self.steps
        # What about multiple similar energy terms????
        try:
            self.zeeman_K = np.array(system.energy.zeeman.H, dtype='float64')
        except:
            self.zeeman_K = None
        try:
            self.exchange_A =  system.energy.exchange.A
        except:
            self.exchange_A = None
        try:
            self.dmi_D = system.energy.dmi.D
            self.Dtype = system.energy.dmi.crystalclass
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
        
    def simulate(self, N: int):
        try:
            os.mkdir(self.schedule_name)
        except:
            pass

        if self.schedule_type == 'FC':
            for i in range(self.steps):                
                self.grid = driver_numba(N, self.grid, self.zeeman_K, self.anisotropic_K, self.anisotropic_u, self.exchange_A, self.dmi_D, self.Ms, self.dx, self.dy, self.dz, self.temperature)
                self.temperature += self.dt
                self.system.m.array = self.grid
                self.save_state(i)
    
    def save_state(self, step: int):
        # save the state inside the schedule folder
        # self.system.m.write(os.path.join(self.schedule_name, f'{step}.ovf'), overwrite=True)
        self.system.m.plane('z').mpl(filename=f'{self.schedule_name}/S_{step}_T_{self.temperature}_Z.png')
        self.system.m.plane('x').mpl(filename=f'{self.schedule_name}/S_{step}_T_{self.temperature}_X.png')


# class Routine:
#     def __init__(self, project_name:str, states: tuple) -> None:
#         #create a folder with the project name
#         self.project_name = project_name
#         self.states = states
#         #create a generator object of Grids
#         self.grids = (Grid(state) for state in states)




        