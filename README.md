# McPy - A Micromagnetic Monte Carlo Simulation package

This has been developed as part of my Master's thesis "Monte Carlo Simulations: Probing the Thermal Stability of Bloch Points" to develop a monte carlo driver to be used as an extension of [Ubermag](https://ubermag.github.io/index.html), a micromagnetic simulations package. Ubermag is a collection of several independent Python packages that can be used independently as well as in combination to be used for other physics simulations such as Fluid Dynamics.

McPy is a package developed for monte carlo simulations in micromagnetics to study magnetic pseudoparticles. The primary aim is to study the thermal stability of bloch points to access their feasibility for information storage. Ubermag already contains well maintained energy minimisation solvers but the goal of this package is to extend the capabilities of Ubermag by adding a non-pertubative approach to energy minimisation that has not been implemented till now.

#### Why Monte Carlo in micromagnetics?

Here are several papers validating the effectiveness of monte carlo in this field
 - [Simulating anti-skyrmions on a lattice](https://www.nature.com/articles/s41598-022-22043-0)
 - [The skyrmion lattice phase in three dimensional chiral magnets from Monte Carlo simulations](https://arxiv.org/abs/1304.6580)


## Installation:

### Requirements
        - Python==3.10.11
        - Ubermag with default oommfc driver : Follow the installation guide at [Ubermag Installation](https://ubermag.github.io/installation.html)
        - NumPy==1.24.3
        - Numba==0.57.0
        - cupy-cuda12x==12.1.0

Installation of above packages can be achieved by running the following command

```bash 

conda env create -f environment.yml

```

## Usage:

#### Visualising a magnetic singularity- Bloch Point

1. Importing Packages

```python
import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
from mcpy.system import MCDriver

```

2. Defining the System parameters
```python
# Magnetisation
Ms = 3.84e5

# Exchange energy constant
A = 8.78e-12

# System geometry
d = 125e-9
hb = 20e-9
ht = 12.5e-9

# Cell discretisation
cell = (5e-9, 5e-9, 2.5e-9)

# Bilayer disk
D_bloch = {'r1': -1.58e-3, 'r2': 1.58e-3, "r1:r2": 1.58e-9}

subregions = {'r1': df.Region(p1=(-d/2, -d/2, -hb), p2=(d/2, d/2, 0)), 'r2': df.Region(p1=(-d/2, -d/2, 0), p2=(d/2, d/2, ht))}

p1 = (-d/2, -d/2, -hb)
p2 = (d/2, d/2, ht)
```
3. Creating Mesh and assigning Energy terms
```python
# Creating mesh
mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

def Ms_fun(point):
        x, y, z = point
        if x**2 + y**2 < (d/2)**2:
                return Ms
        else:
                return 0

system = mm.System(name='bloch_point')

system.energy = mm.Exchange(A=A) + mm.DMI(D=D_bloch, crystalclass='T')
system.m = df.Field(mesh, dim=3, value=(0, 0, 1), norm=Ms_fun)

```

4. Visualing the system

```python

system.m.plane('x').mpl()
```
![Bi-layer disk](images/bilayer.png)

5. Monte Carlo Simulation

```python

# optinal argument for annealing schedule
schedule = schedule={'type': 'FC', 'start_temp': 60, 'end_temp': 0.001, 'steps': 20}

# Defining monte carlo driver object
mc = MCDriver(system, schedule_name='bloch_point', schedule=schedule)

# 10 million Monte Carlo iterations
mc.drive(N=10000000)

# Visualising the system
system.m.plane('x').mpl()

```
![Bloch point](images/bloch_point.png)




## Tests:
To run the automated tests

```bash

python -m unittest tests.py

```

'''

## Guide:

## Repository structure
- `mcpy`: The folder containing the main package
    - `system.py`: MCDriver and Grid classes to initial the Monte Carlo driver object and Grid object
    - `driver.py`: Python function to run the Monte Carlo simulations
    - `energies`
        - `numpy_energies.py`: Numpy optimised energy calculations
        - `numba_energies.py`: Numba optimised energy calculations
- `tests.py`: Unittests
- `Notebooks`: 
     - `Instructios.ipynb`: Instructions on how to use the module
     - `Curie_temperature.ipynb`: Curie temperature calculations
     - `bloch_point.ipynb`: Bloch point simulation
   
<!-- ![Project Structure](images/project%20structure.png) -->

## Documentation
You can find the documentation at `docs/html/index.html`. 

## Outcomes

The full outcomes and analysis can be found in the 
final report of this project in `reports`.

## License:
The scripts and documentation in this project are released under 
the [MIT License](https://github.com/actions/upload-artifact/blob/main/LICENSE)




