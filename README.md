# Pyregress/GPI:

This project provides a suite of Bayesian methods for regression and interpolation in python - specifically using Gaussian-processes. Gaussian process methods are one of the few tools that require little additional complexity for generalization to multiple dimensions without a grid. They also offer the unique ability for non-parametric regression (meaning there is no need to pre-specify basis functions). This tool is intended to be flexible in many regards:

* Use one tool for either interpolation or regression,
* The independent variables may be in one or multiple dimensions,
* Independent variables can be automatically pre-scaled,
* Dependent variables can be automatically transformed,
* Use a variety of predefined kernels - or specified your own,
* Combine kernels with `+` or `*` operators,
* Length-scale parameters in kernels may be universal or unique to each dimension,
* Any kernel parameter may be known or uncertain,
* Optimizes the uncertain kernel parameters (in their natural space or a transformed one),
* Non-parametric approach can be combined with pre-specified basis functions,
* Infer only function values or additional gradient information,
* There is also a builtin utility for model self validation - using leave-one-out analysis.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python 3.8 is highly recommended, and is currently required on the master branch.
Prerequisite libraries include only `numpy`, `scipy` & `numba`, but `matplotlib` is needed for the examples and `pytest` is required for development.

### Installing

Start by simply cloning the repository:
```bash
$ git clone https://github.com/smithseant/pyregress.git pyregress_project
```
Then install locally
```bash
$ cd pyregress_project
$ python -m venv .venv
$ source .venv/bin/activate
$ python -m pip install -e '.[plot,dev]'
```
...depending on your preferred options.

### Running the demonstration

The basic examples are in the documentation of `GPI.__init__` and in the `if __name__ == "__main__":` block of `pyregress/lin_regress.py` and `pyregress/gaussian_processes/gp.py` (but should be run from the project directory):
```bash
$ python -m pyregress.lin_regress
$ python -m pyregress.gaussian_processes.gp
```

### Running the tests (development)

```bash
$ python -m pytest -q tests/test_PolySet.py
$ python -m pytest -q tests/test_GPI.py
```

## Deployment

As for most utility modules it is preferable to reference this project as a prerequisite to your projects rather than embedding it directly.

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Sean T. Smith** - *Initial work* - 
* **Benjamin B. Schroeder** - *Heavy lifting* -

## License
GPLv3.0

## Acknowledgments

Work was partially paid for by the DOE - NNSA - PPSAP II program - CCMSC
