import numpy as np
import micromagneticmodel as mm
from montecarlo.Energies.numpy_energies import zeeman_energy, anisotropic_energy, exchange_energy, dmi_energy, numpy_total
from montecarlo.Driver import driver_numba
from montecarlo.Energies.numba_energies import numba_total, delta_energy, delta_energy2, delta_energy3
import os
        

class Grid:
    def __init__(self, system: mm.System, Ms: float, regions: dict=None):
        """Initializes the system with the given parameters."""
        self.system = self.validate_system(system)
        self.regions = regions
        self.Ms = Ms
        self.grid = self.normalise_grid(system)
        self.dx, self.dy, self.dz = np.array(system.m.mesh.cell, dtype='float64')
        self.zeeman_K = self.get_attribute(system.energy, 'zeeman.H')
        self.exchange_A = self.get_attribute(system.energy, 'exchange.A')
        self.dmi_D = self.initialize_dmi(system.energy.dmi)
        self.Dtype = self.get_attribute(system.energy, 'dmi.crystalclass')
        self.anisotropic_K, self.anisotropic_u = self.initialize_anisotropy(system.energy)
        self.demag_N = self.get_attribute(system.energy, 'demag.N')
    
    def validate_system(self, system):
        """Validates the system."""
        if not isinstance(system, mm.System):
            raise TypeError('system must be a micromagneticmodel.System object.')
        else:
            return system

    def get_regions(self, system: mm.System):
        """Returns the regions from the system or None if not available."""
        try:
            return system.regions
        except AttributeError:
            return None

    def normalise_grid(self, system: mm.System):
        """Computes and returns the grid for the system."""
        magnitudes = np.linalg.norm(system.m.array, axis=-1)
        magnitudes[magnitudes == 0] = 1
        return system.m.array / magnitudes[..., np.newaxis]


    def get_attribute(self, object, attribute: str, default=None):
        """Helper function to get an attribute from an object, returning a default value if not found."""
        try:
            return np.array(eval(f'object.{attribute}'))
        except AttributeError:
            return default

    def initialize_dmi(self, dmi):
        """Creates the DMI array based on regions."""
        try:
            if isinstance(dmi.D, float):
                dmi_D = np.ones(self.grid.shape[:3]) * dmi.D
                dmi_D = np.pad(dmi_D, ((1, 1), (1, 1), (1, 1)), mode='edge')
            else:
                
                dmi_D = np.zeros(self.grid.shape[:3], dtype='float64')
                offset = abs(self.regions['r1'][0])

                r1_start = int((self.regions['r1'][0] + offset)/self.dz)
                r1_end = int((self.regions['r1'][1] + offset)/self.dz)

                r1r2_start = int((self.regions['r1:r2'][0] + offset)/self.dz)

                r2_start = int((self.regions['r2'][0] + offset)/self.dz) + 1
                r2_end = int((self.regions['r2'][1] + offset)/self.dz) + 1

                dmi_D[:, :, r1_start:r1_end] = dmi.D['r1']
                dmi_D[:, :, r1r2_start] = dmi.D['r1:r2']
                dmi_D[:, :, r2_start:r2_end] = dmi.D['r2']
                dmi_D = np.pad(dmi_D, ((1, 1), (1, 1), (1, 1)), mode='edge')
            return dmi_D
            
        except AttributeError:
            return None

    def initialize_anisotropy(self, energy):
        """Initializes the anisotropic energy terms."""
        try:
            return energy.uniaxialanisotropy.K, np.array(energy.uniaxialanisotropy.u)
        except AttributeError:
            return None, None

    def zeeman_energy(self):
        """Computes the Zeeman energy"""
        return zeeman_energy(self.grid, self.zeeman_K, self.Ms, self.dx, self.dy, self.dz)
    
    def anisotropic_energy(self):
        """Computes the anisotropic energy."""
        return anisotropic_energy(self.grid, self.anisotropic_K, self.anisotropic_u, self.dx, self.dy, self.dz)
    
    def exchange_energy(self):
        """Computes the exchange energy."""
        return exchange_energy(self.grid, self.exchange_A, self.dx, self.dy, self.dz)
    
    def dmi_energy(self):
        """Computes the DMI energy."""
        return dmi_energy(self.grid, self.dmi_D, self.Dtype, self.dx, self.dy, self.dz)
    
    def total_energy(self):
        """Computes the total energy."""
        return self.zeeeman_energy() + self.anisotropic_energy() + self.exchange_energy() + self.dmi_energy()
    
    def plot(self, direction: str='z'):
        """Plots the magnetisation in the given direction."""
        self.system.m.plane(direction).mpl()
    
#schedule = {'type': 'FC', 'start_temp': 10.01, 'end_temp': 0.01, 'start_field': [0, 0, 1], 'end_field': [0, 0, -1], steps': 5}

class MCDrive:
    def __init__(self, grid: Grid, energy_calc: int=3, schedule_name: str=None, schedule: dict=None):
        self.grid = grid
        self.schedule_name = schedule_name
        self.schedule = schedule
        self.energy_calc = energy_calc # if energy_calculator == '1' else delta_energy2

        self.initialize_schedule(schedule)

    def initialize_schedule(self, schedule: dict):
        """Initializes the scheduling parameters based on the provided schedule."""
        # Default values
        self.schedule_type = None
        self.temperature = 0.01
        self.field = getattr(self.grid, 'zeeman_K', np.array([0.0, 0.0, 0.0]))
        self.steps = 1
        self.dt = 0.0
        self.df = np.array([0.0, 0.0, 0.0])

        if schedule is None:
            return

        self.schedule_type = schedule.get('type', None)
        if self.schedule_type is None:
            raise KeyError('Schedule type not specified.')

        if 'start_temp' not in schedule:
            raise KeyError('start_temp not specified.')
        self.temperature = schedule['start_temp']

        if 'end_temp' not in schedule:
            raise KeyError('end_temp not specified.')
        self.end_temp = schedule['end_temp']

        if 'steps' not in schedule:
            raise KeyError('steps not specified.')
        self.steps = schedule['steps']
        self.dt = (self.end_temp - self.temperature) / self.steps

        if self.schedule_type == 'FC':
            self.field = schedule.get('field', self.field)
        elif self.schedule_type in ['ZFC', 'HFC']:
            if 'end_field' not in schedule:
                raise KeyError('end_field not specified.')
            self.end_field = schedule['end_field']
            if self.schedule_type == 'ZFC':
                self.field = np.array([0.0, 0.0, 0.0])
            if self.schedule_type == 'HFC':
                self.field = schedule.get('start_field', self.field)
            self.df = (np.array(schedule['end_field']) - np.array(self.field)) / self.steps


        self.log_schedule()

    def log_schedule(self):
        print(f'Schedule initialized: {self.schedule_type}')
        print(f'Start temperature: {self.temperature}')
        print(f'Steps: {self.steps}')
        print(f'End temperature will be {self.temperature + self.steps * self.dt} in {self.dt} per steps.')
        print(f'Start field: {self.field}')
        print(f'End field will be: {self.field + self.steps * self.df} in {self.df} per steps.')

                

    def Drive(self, N: int=5000000, save: bool=False, plot_x: bool=False, plot_y: bool=False, plot_z: bool=False):
        """Runs the simulation for N steps."""
        
        self.save = save
        self.plot_x = plot_x
        self.plot_y = plot_y
        self.plot_z = plot_z

        if self.energy_calc == 1:
            energy_func = numpy_total
        elif self.energy_calc == 2:
            energy_func = numba_total
        elif self.energy_calc == 3:
            energy_func = delta_energy
        elif self.energy_calc == 4:
            energy_func = delta_energy2
        else:
            energy_func = delta_energy3
        
        if self.schedule_name is not None:
            os.makedirs(self.schedule_name, exist_ok=True) # Note: corrected from 'makedir' to 'makedirs'
        
        for i in range(self.steps):                
            self.grid.grid = driver_numba(N, self.grid.grid, energy_func, self.field, self.grid.anisotropic_K, self.grid.anisotropic_u, self.grid.exchange_A, 
                                            self.grid.dmi_D, self.grid.Ms, self.grid.dx, self.grid.dy, self.grid.dz, self.temperature)
            self.temperature += self.dt
            self.field += self.df
            print(f'Step: {i}, Temperature: {round(self.temperature, 2)}, Field: {round(self.field[2], 2)}')
            self.grid.system.m.array = self.grid.grid
            if save:
                self.save_state(i)


    
    def save_state(self, step: int):
        """Saves the state of the system."""

        base_path = f'{self.schedule_name}/' if self.schedule_name else ''
        
        directions_flags = {'z': self.plot_z, 'x': self.plot_x, 'y': self.plot_y}


        for direction, should_plot in directions_flags.items():
            if should_plot:
                file_name = f'{base_path}S_{step}_T_{round(self.temperature, 2)}_F_{round(self.field[2], 2)}_{direction.upper()}.png'
                self.grid.system.m.plane(direction).mpl(filename=file_name)

    


# class Routine:
#     def __init__(self, project_name:str, states: tuple) -> None:
#         #create a folder with the project name
#         self.project_name = project_name
#         self.states = states
#         #create a generator object of Grids
#         self.grids = (Grid(state) for state in states)