# Project Title

This project provides a suite of Bayesian methods for regression and
interpolation - specifically using Gaussian-processes in python. Gaussian
process methods are one of the few tools that require little additional
complexity to go into multiple dimensions without a grid. They also offer
the unique ability for non-parametric (meaning there is no need to pre-specify
basis functions) regression. This tool is intended to be flexible in many
regards:
- One or multiple dimensions may be used for the independent variables,
- Independent variables can be automatically pre-scaled,
- Use a variety of predefined kernels or specified your own,
- Combine kernels with '+' or '*' operators,
- Length-scale parameters in kernels may be unique to each dimension,
- Any kernel parameter may be known or uncertain,
- Non-parametric approach can be combined with pre-specified basis functions,
- Use automatic transformations that make the method more widely applicable,
- Specify and infer slope (or gradient) information,
There is also a utility for model self validation - using leave-one-out style
analysis.

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes
on how to deploy the project on a live system.

### Prerequisites

Python 2.7 or 3.5 are both supported - on separate branches. Prerequisites
include only numpy & scipy, but matplotlib is needed for the examples.
Anaconda provides all of these by default.

### Installing

No special steps are required to install the module - simply clone the repository
```buildoutcfg
git clone git@bitbucket.org:team_sean/pyregress.git
```

### Running the tests

The basic examples are in the if-main of pyregress0.py:
```buildoutcfg
$ python pyregress0
```

## Deployment

As for most utility projects it is preferable to reference this project as a
prerequisite to your projects rather than embedding it directly.

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Sean T. Smith** - *Initial work* - 
* **Benjamin B. Schroeder** - *Heavy lifting* -

## License

Be aware: this project licensing has not yet been decided.

## Acknowledgments

Work was paid for by the DOE - NNSA - PPSAP II program - CCMSC