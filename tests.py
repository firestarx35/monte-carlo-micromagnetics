import unittest
import numpy as np
import os

from mcpy.system import Grid, MCDriver
import mcpy.energies.numpy_energies as np_energies
import mcpy.energies.numba_energies as nb_energies
from mcpy.driver import driver_numpy, driver_numba

import oommfc as oc
import discretisedfield as df
import micromagneticmodel as mm


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.region = df.Region(
            p1=(-50e-9, -50e-9, -10e-9), p2=(50e-9, 50e-9, 10e-9))
        self.mesh = df.Mesh(region=self.region, cell=(2.5e-9, 2.5e-9, 2.5e-9))
        self.system = mm.System(name='test')
        self.system.energy = (mm.Exchange(A=1.6e-11) + mm.DMI(D=4e-3, crystalclass='D2d_z') +
                              mm.UniaxialAnisotropy(K=0.51e6, u=(0, 0, 1)) + mm.Zeeman(H=(0, 0, 2e5)))
        self.system.m = df.Field(self.mesh, dim=3, value=(0, 0, 1), norm=1.1e6)


class TestGridFunctions(BaseTestCase):
    def test_init(self):
        test_grid = Grid(self.system)
        self.assertEqual(test_grid.grid.shape,
                         self.system.m.array.shape, "Grid shape mismatch")
        self.assertEqual(
            test_grid.dx, self.system.m.mesh.cell[0], "dx value mismatch")
        self.assertEqual(
            test_grid.dy, self.system.m.mesh.cell[1], "dy value mismatch")
        self.assertEqual(
            test_grid.dz, self.system.m.mesh.cell[2], "dz value mismatch")
        self.assertEqual(type(test_grid.system), type(
            self.system), "System type mismatch")
        self.assertTrue(np.array_equal(test_grid.zeeman_H,
                        np.array(self.system.energy.zeeman.H)))
        self.assertEqual(
            test_grid.exchange_A, self.system.energy.exchange.A, "Exchange A value mismatch")
        self.assertEqual(
            np.unique(test_grid.dmi_D), self.system.energy.dmi.D, "DMI D value mismatch")
        self.assertEqual(
            test_grid.Dtype, self.system.energy.dmi.crystalclass, "DMI Dtype mismatch")
        self.assertEqual(test_grid.anisotropy_K,
                         self.system.energy.uniaxialanisotropy.K, "Anisotropy K mismatch")
        self.assertIsInstance(test_grid.anisotropy_u,
                              np.ndarray, "Anisotropy u mismatch")

    def test_validate(self):
        test_grid = Grid(self.system)
        self.assertIsInstance(
            test_grid._validate_system(self.system), mm.System)

        with self.assertRaises(TypeError, msg='system must be a micromagneticmodel.System object.'):
            test_grid._validate_system(1)

    def test_normalise_grid(self):
        test_grid = Grid(self.system)
        test_data = self.system.m.array
        mags = np.linalg.norm(test_data, axis=-1)
        mags[mags == 0] = 1
        test_data = test_data / mags[..., np.newaxis]
        self.assertTrue(np.array_equal
                        (test_grid._normalise_grid(self.system), test_data))

    def test_get_attribute(self):
        test_grid = Grid(self.system)
        self.assertTrue(np.array_equal(test_grid._get_attribute(
            self.system.energy, 'zeeman.H'), np.array(self.system.energy.zeeman.H)))
        self.assertEqual(test_grid._get_attribute(self.system, 'test'), None)

    def test_get_initialise_dmi(self):
        test_grid = Grid(self.system)
        D = self.system.energy.dmi.D
        dmi_D = np.ones(self.system.m.array.shape[:3])
        dmi_D = np.pad(dmi_D, ((1, 1), (1, 1), (1, 1)), mode='edge') * D
        # self.assertTrue(np.array_equal(test_grid._initialise_dmi(
        #     self.system.energy.dmi), dmi_D))
        self.assertEqual(test_grid._initialise_dmi(None), None)

    def test_initialise_anisotropy(self):
        test_grid = Grid(self.system)
        K = np.array(self.system.energy.uniaxialanisotropy.K)
        u = np.array(self.system.energy.uniaxialanisotropy.u)
        test_K, test_u = test_grid._initialise_anisotropy(
            self.system.energy)
        self.assertEqual(np.array(K),  test_K)
        self.assertTrue(np.array_equal(u, test_u))
        self.assertEqual(test_grid._initialise_anisotropy(None), (None, None))

    def test_energy(self):
        test_grid = Grid(self.system)
        self.assertAlmostEqual(test_grid.zeeman_energy(), oc.compute(
            self.system.energy.zeeman.energy, self.system), places=17)
        self.assertAlmostEqual(test_grid.exchange_energy(), oc.compute(
            self.system.energy.exchange.energy, self.system), places=17)
        self.assertAlmostEqual(test_grid.dmi_energy(), oc.compute(
            self.system.energy.dmi.energy, self.system), places=17)
        self.assertAlmostEqual(test_grid.anisotropy_energy(), oc.compute(
            self.system.energy.uniaxialanisotropy.energy, self.system), places=17)
        self.assertAlmostEqual(test_grid.total_energy(), oc.compute(
            self.system.energy.energy, self.system), places=17)


class TestMCDriver(BaseTestCase):
    def test_init(self):
        test_driver = MCDriver(
            system=self.system, energy_calc='numba', schedule_name='test_schedule')
        self.assertIsInstance(test_driver.grid, Grid)
        self.assertIsInstance(test_driver.grid.system, mm.System)
        self.assertEqual(test_driver.schedule_name, 'test_schedule')
        self.assertEqual(test_driver.energy_calc, 'numba')

    def test_initialise_schedule(self):
        schedule = {'start_temp': 0.00001, 'end_temp': 100, 'start_field': [
            0, 0, 100], 'end_field': [0, 0, 20], 'steps': 1000}
        test_driver = MCDriver(
            system=self.system, energy_calc='numba', schedule_name='test_schedule')

        with self.assertRaises(KeyError, msg='Schedule type not specified.'):
            test_driver._initialise_schedule(schedule=schedule)

        schedule = {'type': 'MC', 'start_temp': 0.00001, 'end_temp': 100, 'start_field': [
            0, 0, 100], 'end_field': [0, 0, 20], 'steps': 1000}

        with self.assertRaises(KeyError, msg='Schedule type must be one of the following: FC, ZFC, HFC.'):
            test_driver._initialise_schedule(schedule=schedule)

        schedule = {'type': 'FC', 'start_temp': 0.00001, 'start_field': [
            0, 0, 100], 'end_field': [0, 0, 20], 'steps': 1000}
        with self.assertRaises(KeyError, msg='end_temp not specified.'):
            test_driver._initialise_schedule(schedule=schedule)

        schedule = {'type': 'ZFC', 'start_field': [
            0, 0, 100], 'end_field': [0, 0, 20], 'steps': 1000}
        with self.assertRaises(KeyError, msg='start_temp not specified.'):
            test_driver._initialise_schedule(schedule=schedule)

        schedule = {'type': 'HFC', 'start_temp': 0.00001,
                    'end_temp': 100, 'end_field': [0, 0, 100], 'steps': 1000}
        with self.assertRaises(KeyError, msg='start_field not specified.'):
            test_driver._initialise_schedule(schedule=schedule)

        schedule = {'type': 'ZFC', 'start_temp': 0.00001,
                    'end_temp': 100, 'start_field': [0, 0, 100], 'steps': 1000}
        with self.assertRaises(KeyError, msg='end_field not specified.'):
            test_driver._initialise_schedule(schedule=schedule)

    # def test_drive(self):
    #     test_driver = MCDriver(
    #         system=self.system, energy_calc='numpy', schedule_name='test_schedule')
    #     test_driver.drive(N=2, save=True, plot_x=True,
    #                       plot_y=True, plot_z=True)
    #     # check if a folder named 'test_schedule' is created in the current directory and if the plots named
    #     # 'S_0_T_0.00001_F_200000_X.png', 'S_0_T_0.00001_F_200000_Y.png' and 'S_0_T_0.00001_F_200000_Z.png' are created in the folder
    #     # Check if a folder named 'test_schedule' is created in the current directory
    #     self.assertTrue(os.path.exists('test_schedule'))

    #     expected_files = [
    #         'S_0_T_0.0_F_200000.0_X.png',
    #         'S_0_T_0.0_F_200000.0_Y.png',
    #         'S_0_T_0.0_F_200000.0_Z.png'
    #     ]
    #     for fname in expected_files:
    #         self.assertTrue(os.path.exists(
    #             os.path.join('test_schedule', fname)))

    #     self.tearDown()

    # def tearDown(self):
    #     # Cleanup after each test
    #     if os.path.exists('test_schedule'):
    #         # Remove the created files
    #         for fname in os.listdir('test_schedule'):
    #             os.remove(os.path.join('test_schedule', fname))
    #         # Remove the directory
    #         os.rmdir('test_schedule')


class TestNumpyEnergiesFunctions(BaseTestCase):
    def test_numpy_zeeman(self):
        test_grid = Grid(self.system)
        self.assertAlmostEqual(np_energies.zeeman_energy(test_grid.grid, test_grid.zeeman_H, test_grid.Ms, test_grid.dx,
                               test_grid.dy, test_grid.dz), oc.compute(self.system.energy.zeeman.energy, self.system), places=17)

    def test_numpy_anisotropy(self):
        test_grid = Grid(self.system)
        self.assertAlmostEqual(np_energies.anisotropy_energy(test_grid.grid, test_grid.anisotropy_K, test_grid.anisotropy_u, test_grid.dx,
                               test_grid.dy, test_grid.dz), oc.compute(self.system.energy.uniaxialanisotropy.energy, self.system), places=17)

    def test_numpy_exchange(self):
        test_grid = Grid(self.system)
        self.assertAlmostEqual(np_energies.exchange_energy(test_grid.grid, test_grid.exchange_A, test_grid.dx,
                               test_grid.dy, test_grid.dz), oc.compute(self.system.energy.exchange.energy, self.system), places=17)

    def test_numpy_dmi(self):
        test_grid = Grid(self.system)
        self.assertAlmostEqual(np_energies.dmi_energy(test_grid.grid, test_grid.Dtype, test_grid.dmi_D, test_grid.dx,
                               test_grid.dy, test_grid.dz), oc.compute(self.system.energy.dmi.energy, self.system), places=17)

    # def test_numpy_delta(self):
        # test_grid = Grid(system)
        # self.assertAlmostEqual(np_energies.numpy_delta(test_)
        # TODO: Implement test_numpy_delta


class TestDriverFunctions(BaseTestCase):

    def test_driver_numpy(self):
        self.testing_grid = Grid(self.system)
        self.zeeman_H = np.array([0, 0, 1])
        self.temperature = 300  # Kelvin
        self.N = 10  # Number of iterations for the Monte Carlo simulation
        magnetization = driver_numpy(
            self.N, self.testing_grid, self.zeeman_H, self.temperature)
        self.assertIsInstance(magnetization, np.ndarray)
        self.assertEqual(magnetization.shape, (self.testing_grid.grid.shape))

    # def test_rotations(self): #TODO: Implement test_rotations
    #     magnetization = driver_numpy(
    #         self.N, self.grid, self.zeeman_H, self.temperature)
    #     self.assertTrue(np.allclose(np.linalg.norm(magnetization, axis=-1), 1))


class TestNumbaEnergiesFunctions(unittest.TestCase):
    # TODO: Match with exact answers

    def test_zeeman_energy_numba(self):
        grid = np.ones((5, 5, 5, 3))
        zeeman_H = np.array([0, 0, 1])
        Ms = 1.1e6
        dx = dy = dz = 2.5e-9
        energy = nb_energies.zeeman_energy(grid, zeeman_H, Ms, dx, dy, dz)
        self.assertIsInstance(energy, float)

    def test_anisotropy_energy_numba(self):
        grid = np.ones((5, 5, 5, 3))
        anisotropy_K = 3
        anisotropy_u = np.array([0, 1, 0])
        dx = dy = dz = 2.5e-9
        energy = nb_energies.anisotropy_energy(
            grid, anisotropy_K, anisotropy_u, dx, dy, dz)
        self.assertIsInstance(energy, float)

    def test_exchange_energy_numba(self):
        grid = np.ones((5, 5, 5, 3))
        A = 1.6e-11
        dx = dy = dz = 2.5e-9
        energy = nb_energies.exchange_energy(grid, A, dx, dy, dz)
        self.assertIsInstance(energy, float)

    def test_dmi_energy_numba(self):
        grid = np.ones((5, 5, 5, 3))
        D = 4e-3 * np.ones((3, 3, 3))
        dx = dy = dz = 2.5e-9
        energy = nb_energies.dmi_Cnvz(grid, D, dx, dy, dz)
        # TODO: Add more types of DMI
        self.assertIsInstance(energy, float)


if __name__ == "__main__":
    unittest.main()
