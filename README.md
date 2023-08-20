# McPy - A Micromagnetic Monte Carlo Simulation package

McPy is a package developed for monte carlo simulations in micromagnetics to study magnetic pseudoparticles. This has been developed as part of my Master's thesis "Parallelisation and Optimisation of Monte Carlo Simulation in Nanomagnetism" to develop an extension of Ubermag(https://ubermag.github.io/index.html), a micromagnetic simulations package. Ubermag is a collection of several independent Python packages that can be used independently as well as in combination to be used for other physics simulations such as Fluid Dynamics.

Ubermag already contains well maintained energy minimisation solvers but the goal of this package is to extend the capabilities of Ubermag by adding a non-pertubative approach to energy minimisation that has not been implemented till now.

#### Why Monte Carlo in micromagnetics?

Here are several papers validating the effectiveness of monte carlo in this field
 - 
 - 
 - 

## Installation:
- ### Requirements
        - Python==3.10.11
        - Ubermag with default oommfc driver : Follow the installation guide at (https://ubermag.github.io/installation.html)
        - NumPy==1.24.3
        - Numba==0.57.0
        - cupy-cuda12x==12.1.0

Installation of abovepackages can be achieved by running the below command

```
conda env create -f environment.yml

```


## Tests:
To run the automated tests

```
python -m unittest montecarlo/test
```

## Guide:

## Repository structure
- `mcpy`: The folder containing the main package
    - `system.py`: Python classes to 
    - `driver.py`: Python function to
    - `energies`
        - `numpy_energies.py`:
        - `numba_energies.py`:
- `tests.py`:
- `Notebooks`:
    - `Energy_validation.ipynb`: 
    - `Energy_validation2.ipynb`:
    - `Simulation_validation.ipynb`:
    - `Simulation_validation2.ipynb`:
    - `Curie_temperature.ipynb`:
    - `bloch_point.ipynb`:
    - `Instructios.ipynb`:

![Project Structure](images/project%20structure.png)

## Documentation
You can find the documentation at `docs/html/index.html`. 


## License:



