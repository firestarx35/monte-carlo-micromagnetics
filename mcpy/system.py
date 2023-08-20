import numpy as np
import micromagneticmodel as mm
from mcpy.energies.numpy_energies import zeeman_energy, anisotropy_energy, exchange_energy, dmi_energy, numpy_delta
from mcpy.driver import driver_numpy, driver_numba
from mcpy.energies.numba_energies import numba_delta, delta_energy, delta_energy2
import os


"""This module contains the Grid and MCDriver classes required for micromagnetic monte carlo simulations."""


class MCDriver:
    def __init__(self, system: mm.System, energy_calc: str = 'numba', schedule_name: str = None, schedule: dict = None) -> None:
        """Monte Carlo Driver object.

        Creates a Monte Carlo Driver object for the given micromagneticmodel.System object. 
        The object contains the Grid object, the schedule and the energy calculation function. 
        The energy calculation function can be either 'numpy', 'numba' or created for specific cases 'delta_energy or delta_energy2'.
        schedule_name is the name of a folder that will be created, and the data generated will be saved inside that folder.
        A schedule can be provided as a dictionary. If no schedule is provided, it defaults to None and no temperature and field changes will occur.

        Example schedules:

        Field cooling (FC): schedules = {'type': 'FC', 'start_temp': 0.00001, 'end_temp': 100, 'start_field': [0, 0, 0], 'steps': 1000}
        Zero field cooling (ZFC): schedules = {'type': 'ZFC', 'start_temp': 0.00001, 'end_temp': 100, 'end_field': [0, 0, 100],  'steps': 1000}
        High field cooling (HFC): schedules = {'type': 'HFC', 'start_temp': 0.00001, 'end_temp': 100, 'start_field': [0, 0, 100], 'end_field': [0, 0, 20],  'steps': 1000}


        Args:
            system (micromagneticmodel.System): micromagneticmodel.System object.
            energy_calc (str, optional): The energy calculation function. Defaults to numba.
            schedule_name (str, optional): The name of the schedule. Defaults to None.
            schedule (dict, optional): The schedule dictionary. Defaults to None.

        Examples:
            >>> import micromagneticmodel as mm
            >>> import discretisedfield as df
            >>> import mcpy
            >>> import numpy as np

            >>> region = df.Region(p1=(-50e-9, -50e-9, -10e-9), p2=(50e-9, 50e-9, 10e-9))
            >>> mesh = df.Mesh(region=region, cell=(2.5e-9, 2.5e-9, 2.5e-9))
            >>> system = mm.System(name='test')
            >>> system.energy = (mm.Exchange(A=1.6e-11) + mm.DMI(D=4e-3, crystalclass='D2d_z') + mm.UniaxialAnisotropy(K=0.51e6, u=(0, 0, 1)) + mm.Zeeman(H=(0, 0, 2e5)))
            >>> system.m = df.Field(mesh, dim=3, value=(0,0,1), norm=1.1e6)
            >>> driver = mcpy.MCDriver(system, energy_calc='numba', schedule_name='test', schedule={'type': 'FC', 'start_temp': 0.00001, 'end_temp': 100, 'start_field': [0, 0, 0], 'steps': 10})
            >>> driver.drive(N=5000000, save=True, plot_x=True, plot_y=True, plot_z=True)
        """

        self.grid = Grid(system)
        self.schedule_name = schedule_name
        self.schedule = schedule
        self.energy_calc = energy_calc  # if energy_calculator == '1' else delta_energy2

        self._initialize_schedule(schedule)

    def _initialize_schedule(self, schedule: dict) -> None:
        """Initializes the scheduling parameters based on the provided schedule.

        Args:
            schedule (dict): The schedule dictionary.

        Raises:
            KeyError: If the schedule type is not specified.
            KeyError: If the start temperature is not specified.
            KeyError: If the end temperature is not specified.
            KeyError: If the number of steps is not specified.

        Returns:
            None
        """
        # default values
        self.schedule_type = None
        self.temperature = 0.00001
        if getattr(self.grid, 'zeeman_H') is not None:
            self.field = getattr(self.grid, 'zeeman_H')
        else:
            self.field = np.array([0.0, 0.0, 0.0])
        self.steps = 0
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
            self.field = np.array(schedule.get('start_field', self.field))
        elif self.schedule_type in ['ZFC', 'HFC']:
            if 'end_field' not in schedule:
                raise KeyError('end_field not specified.')
            self.end_field = schedule['end_field']
            if self.schedule_type == 'ZFC':
                self.field = np.array([0.0, 0.0, 0.0])
            if self.schedule_type == 'HFC':
                self.field = np.array(schedule.get('start_field', self.field))
            self.df = (
                np.array(schedule['end_field']) - self.field) / self.steps

        self.log_schedule()

    def log_schedule(self) -> None:
        """Logs the schedule."""

        print(f'Schedule initialized: {self.schedule_type}')
        print(f'Start temperature: {round(self.temperature, 2)}K')
        print(f'Steps: {self.steps}')
        print(
            f'End temperature will be {round(self.temperature + self.steps * self.dt, 2)}K in {round(self.dt, 3)}K per steps.')
        print(f'Start field: {self.field[2]} A/m')
        print(
            f'End field will be: {round((self.field + self.steps * self.df)[2], 2)} in {round(self.df[2], 2)} A/m per steps.')

    def drive(self, N: int = 5000000, save: bool = False, plot_x: bool = False, plot_y: bool = False, plot_z: bool = False) -> None:
        """Runs the micromagnetic monte carlo simulation for N steps using the Grid object. And follows the schedule if provided.
        Plots the magnetisation in the specified planes if the flags are set to True.

        Args:
            N (int, optional): Number of steps. Defaults to 5000000.
            save (bool, optional): Whether to save the results. Defaults to False.
            plot_x (bool, optional): Plot the x-component of the magnetisation. Defaults to False.
            plot_y (bool, optional): Plot the y-component of the magnetisation. Defaults to False.
            plot_z (bool, optional): Plot the z-component of the magnetisation. Defaults to False.

        Returns:
            None
        """

        self.save = save
        self.plot_x = plot_x
        self.plot_y = plot_y
        self.plot_z = plot_z

        if self.energy_calc == 'numpy':
            energy_func = numpy_delta
        elif self.energy_calc == 'numba':
            energy_func = numba_delta
        # TODO: Remove this (Created for specific study purposes)
        elif self.energy_calc == 'delta_energy':
            energy_func = delta_energy
        # TODO: Remove this (Created for specific study purposes)
        elif self.energy_calc == 'delta_energy2':
            energy_func = delta_energy2
        else:
            energy_func = numba_delta

        if self.schedule_name is not None:
            # create a folder with the schedule name
            os.makedirs(self.schedule_name, exist_ok=True)

        for i in range(self.steps + 1):
            print(
                f'Step: {i}, Temperature: {round(self.temperature, 2)}K, Field: {round(self.field[2], 2)} A/m')
            if self.energy_calc == 'numpy':
                self.grid.grid = driver_numpy(
                    N, self.grid, self.field, self.temperature)
            else:
                self.grid.grid = driver_numba(N, self.grid.grid, energy_func, self.field, self.grid.anisotropy_K, self.grid.anisotropy_u,
                                              self.grid.exchange_A, self.grid.dmi_D, str(self.grid.Dtype), self.grid.Ms, self.grid.dx, self.grid.dy, self.grid.dz, self.temperature)

            self.grid.system.m.array = self.grid.grid
            if save:
                self._save_state(i)
            self.temperature += self.dt
            # Done to avoid data type errors when adding float and int
            np.add(self.field, self.df, out=self.field,
                   casting='unsafe')  # self.field += self.df

    def _save_state(self, step: int) -> None:
        """Saves the state/plot of the system. Based on direction flags provided in MCDriver.driver() function,
            the state is saved in the specified planes. The state is saved in a file with the following format:
            S_{step}_T_{temperature}_F_{field}.png

        Args:
            step (int): The step number.

        Returns:
            None
        """

        base_path = f'{self.schedule_name}/' if self.schedule_name else ''

        directions_flags = {'z': self.plot_z,
                            'x': self.plot_x, 'y': self.plot_y}

        for direction, should_plot in directions_flags.items():
            if should_plot:
                file_name = f'{base_path}S_{step}_T_{round(self.temperature, 2)}_F_{round(self.field[2], 2)}_{direction.upper()}.png'
                self.grid.system.m.plane(direction).mpl(filename=file_name)


class Grid:
    def __init__(self, system: mm.System) -> None:
        """Grid object required by mcpy for micromagnetic simulations. This class contains all the parameters used by the MCDriver.  
        Created by abstracting and converting the required parameters from a micromagneticmodel.System object. This normalises the grid and creates the DMI array based on regions.
        and converts other parameters to numpy arrays.

        Args:
            system (micromagneticmodel.System): The system to be initialized.

        Raises:
            TypeError: If the system is not a micromagneticmodel.System object.

        Examples:
            >>> import micromagneticmodel as mm
            >>> import discretisedfield as df
            >>> import mcpy
            >>> import numpy as np

            >>> region = df.Region(p1=(-50e-9, -50e-9, -10e-9), p2=(50e-9, 50e-9, 10e-9))
            >>> mesh = df.Mesh(region=region, cell=(2.5e-9, 2.5e-9, 2.5e-9))
            >>> system = mm.System(name='test')
            >>> system.energy = (mm.Exchange(A=1.6e-11) + mm.DMI(D=4e-3, crystalclass='D2d_z') + mm.UniaxialAnisotropy(K=0.51e6, u=(0, 0, 1)) + mm.Zeeman(H=(0, 0, 2e5)))
            >>> system.m = df.Field(mesh, dim=3, value=(0,0,1), norm=1.1e6)
            >>> gd = mcpy.Grid(system)
            >>> np.array_equal(gd.grid, system.m.array)
            True
        """
        self.system = self._validate_system(system)
        self.Ms = np.unique(system.m.norm.value).max()
        self.grid = self._normalise_grid(system)
        self.dx, self.dy, self.dz = np.array(
            system.m.mesh.cell, dtype='float64')
        self.zeeman_H = self._get_attribute(system.energy, 'zeeman.H')
        self.exchange_A = self._get_attribute(system.energy, 'exchange.A')
        self.dmi_D = self._initialize_dmi(system.energy.dmi)
        self.Dtype = self._get_attribute(system.energy, 'dmi.crystalclass')
        self.anisotropy_K, self.anisotropy_u = self._initialize_anisotropy(
            system.energy)
        self.demag_N = self._get_attribute(system.energy, 'demag.N')

    def _validate_system(self, system) -> mm.System:
        """Validates the micromagneticmodel.System obbject.

        Args:
            system (micromagneticmodel.system): The system to validate.

        Raises:
            TypeError: If the system is not a micromagneticmodel.System object.

        Returns:
            system (micromagneticmodel.System): The validated system.
        """
        if not isinstance(system, mm.System):
            raise TypeError(
                'system must be a micromagneticmodel.System object.')
        else:
            return system

    def _normalise_grid(self, system: mm.System) -> np.ndarray:
        """Normalises the grid for the micromagnetic monte carlo simulations.

        Args:
            system (mm.System): The system to normalise the grid for.

        Returns:
            grid (mm.Field): The Normalised grid.
        """
        magnitudes = np.linalg.norm(system.m.array, axis=-1)
        magnitudes[magnitudes == 0] = 1
        return system.m.array / magnitudes[..., np.newaxis]

    def _get_attribute(self, object, attribute: str, default=None) -> object:
        """Helper function to get an attribute from an object, returning a default value if not found.

        Args:
            object (object): The object to get the attribute from.
            attribute (str): The attribute to get.
            default (object): The default value to return if the attribute is not found.

        Raises:
            AttributeError: If the attribute is not found and no default value is given.

        Returns:
            The attribute if found, otherwise the default value.
        """
        try:
            attri = np.array(eval(f'object.{attribute}'))
            print(f'{attribute} found.')
            return attri
        except AttributeError:
            print(f'{attribute} not found.')
            return default

    def _initialize_dmi(self, dmi) -> np.ndarray:
        """Creates the DMI array based on regions. This currently supports subregions along z-axis only.

        If the DMI is a float, then the array is filled with that value.
        If the DMI is a dictionary, then the array is filled with the values in the dictionary according to regions.
        The array is padded with zeros for boundary conditions.

        Args:
            dmi (float or dict): The DMI value(s) to be used.

        Returns:
            dmi_D (np.ndarray): The DMI array.
        """
        try:
            if isinstance(dmi.D, float):
                dmi_D = np.ones(self.grid.shape[:3]) * dmi.D
                # padding to avoid boundary conditions
                dmi_D = np.pad(dmi_D, ((1, 1), (1, 1), (1, 1)), mode='edge')
            else:
                dmi_D = np.zeros(self.grid.shape[:3])
                offset = 0
                for region in self.system.m.mesh.subregions.keys():
                    p1, p2 = np.array(self.system.m.mesh.subregions[region].p1), np.array(
                        self.system.m.mesh.subregions[region].p2)
                    discretised = (p2 - p1) / np.array(self.system.m.mesh.cell)
                    region_size = int(discretised[2])
                    dmi_D[:, :, offset:offset +
                          region_size] = self.system.energy.dmi.D[region]
                    offset += region_size
                dmi_D = np.pad(dmi_D, ((1, 1), (1, 1), (1, 1)), mode='edge')
            return dmi_D

        except AttributeError:
            print('DMI not found.')
            return None

    def _initialize_anisotropy(self, energy) -> tuple:
        """Initializes the anisotropic energy terms.

        Args:
            energy (micromagneticmodel.Energy): The energy object to get the anisotropy from.

        Returns:
            un_K (float): The anisotropy constant.
        """
        try:
            uan_K = np.array(energy.uniaxialanisotropy.K)
            uan_u = np.array(energy.uniaxialanisotropy.u)
            print('Anisotropy found.')
            return uan_K, uan_u

        except AttributeError:
            print('Anisotropy not found.')
            return None, None

    def zeeman_energy(self) -> float:
        """Computes the Zeeman energy
        Returns:
            energy (float): Zeeman energy of the system
        """
        return zeeman_energy(self.grid, self.zeeman_H, self.Ms, self.dx, self.dy, self.dz)

    def anisotropy_energy(self) -> float:
        """Computes the uniaxialanisotropic energy.
        Returns:
            energy (float): Uniaxialanisotropic energy of the system"""
        return anisotropy_energy(self.grid, self.anisotropy_K, self.anisotropy_u, self.dx, self.dy, self.dz)

    def exchange_energy(self) -> float:
        """Computes the exchange energy the system.
        Returns:
            energy (float): Exchange energy of the system
        """
        return exchange_energy(self.grid, self.exchange_A, self.dx, self.dy, self.dz)

    def dmi_energy(self) -> float:
        """Computes the DMI energy of the system.
        Returns:
            energy (float): DMI energy of the system
        """
        return dmi_energy(self.grid, self.Dtype, self.dmi_D, self.dx, self.dy, self.dz)

    def total_energy(self) -> float:
        """Computes the total energy of the system.
        Returns:
            energy (float): Total energy of the system
        """
        return self.zeeeman_energy() + self.anisotropy_energy() + self.exchange_energy() + self.dmi_energy()

    def plot(self, direction: str = 'z') -> None:
        """Plots the magnetisation of the system in the given direction. Directions can be 'x', 'y' or 'z'.

        Args:
            direction (str, optional): The direction to plot. Defaults to 'z'.

        Returns:
            None
        """
        self.system.m.plane(direction).mpl()
