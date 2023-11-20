
# Simple Symbolic Robotics

Simple Symbolic Robotics is a Python library providing a collection of helper functions designed for fundamental robotics computations, particularly in kinematics. This library leverages symbolic computation using SymPy, combined with numerical libraries like NumPy and SciPy, to offer a versatile toolkit for robotics enthusiasts, researchers, and engineers.

## Features

- Symbolic computation of skew-symmetric matrices.
- Functions to validate and compute rotation matrices.
- Utilities for handling homogeneous transformations and rotations.
- Simplified interfaces for common robotics calculations.

## Installation

Install Simple Symbolic Robotics using pip:

```bash
pip install simple_symbolic_robotics
```

## Usage

Import the library and use its functions in your Python scripts or interactive sessions.

```python
import simple_symbolic_robotics as ssr

# Example usage:
R = ssr.Rx(pi/4)  # Rotation about the x-axis by 45 degrees
```

For more use see jupyter notebooks in the [notebooks folder](https://github.com/AdoHaha/simple_symbolic_robotics/tree/master/notebooks):
<a target="_blank" href="https://colab.research.google.com/github/AdoHaha/simple_symbolic_robotics/blob/master/notebooks/basic_examples.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Licensing

The code in this project is licensed under the MIT License.
