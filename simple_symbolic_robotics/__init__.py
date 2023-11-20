#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy.linalg import norm
import scipy
import sympy
from sympy import pi, acos, sin, latex, pprint

# Helper functions
platex = lambda res: print(latex(res))  # Print result as LaTeX
platexN = lambda res: platex(sympy.N(res, 2))  # Print result as rounded LaTeX
pN = lambda res: pprint(sympy.N(res, 2))  # pprint result after rounding to two decimal spaces


def skew_symmetric_matrix(vector: np.ndarray) -> sympy.Matrix:
    """
    Returns the skew-symmetric matrix for a given vector.
    :param vector: A 3-element vector.
    :return: A 3x3 skew-symmetric matrix.
    """
    S = sympy.Matrix([[0, -vector[2], vector[1]],
                      [vector[2], 0, -vector[0]],
                      [-vector[1], vector[0], 0]])
    return S

# Alias for backward compatibility
S_sym = skew_symmetric_matrix


def is_rotation_matrix(matrix: sympy.Matrix) -> bool:
    """
    Checks if a matrix is a rotation matrix.
    :param matrix: A square matrix.
    :return: True if matrix is a rotation matrix, False otherwise.
    """
    return matrix * matrix.T == matrix.T * matrix == sympy.diag(1, 1, 1)


def axis_angle_rotation(axis: np.ndarray, angle: float, print_flag: bool = False) -> sympy.Matrix:
    """
    Returns the rotation matrix for a rotation around a given axis by a given angle.
    :param axis: A 3-element vector representing the rotation axis.
    :param angle: The rotation angle in radians.
    :param print_flag: If True, prints intermediate steps.
    :return: A 3x3 rotation matrix.
    """
    norm_axis = sympy.Matrix(axis) / sympy.Matrix(axis).norm()
    A = skew_symmetric_matrix(norm_axis)
    A_squared = A * A
    result = sympy.eye(3) + sympy.sin(angle) * A + (1 - sympy.cos(angle)) * A_squared

    if print_flag:
        print(f"Axis: {axis}")
        print(f"Normalized Axis: \n{norm_axis}")
        print(f"Skew-symmetric from normalized axis: \n{A}")
        print(f"Square of skew-symmetric: \n{A_squared}")
        print(f"Result: \n{result}")

    return result

# Alias for backward compatibility
os_obrot = axis_angle_rotation


def rotation_to_axis_angle(R: sympy.Matrix) -> tuple:
    """
    Calculates the axis and angle from a rotation matrix.
    :param R: A 3x3 rotation matrix.
    :return: A tuple (axis, angle) where axis is a 3-element vector and angle is in radians.
    """
    M_rot = np.array(R)
    tr = np.trace(M_rot)
    if np.array_equal(M_rot, np.eye(3)):
        theta = 0
        axis = np.zeros((3, 1))
    elif tr == -1:
        theta = sympy.pi
        axis = _compute_axis(M_rot)
    else:
        theta = acos((tr - 1) / 2)
        MM = M_rot - M_rot.T
        axis = (1 / (2 * sin(theta))) * sympy.Matrix([[MM[2, 1]], [MM[0, 2]], [MM[1, 0]]])
    return axis, theta

def _compute_axis(M_rot):
    if M_rot[2, 2] != -1:
        axis = (1 / sympy.sqrt(2 * (1 + M_rot[2, 2]))) * np.array(
            [[M_rot[0, 2]], [M_rot[1, 2]], [1 + M_rot[2, 2]]])
    elif M_rot[1, 1] != -1:
        axis = (1 / sympy.sqrt(2 * (1 + M_rot[1, 1]))) * np.array(
            [[M_rot[0, 1]], [1 + M_rot[1, 1]], [M_rot[2, 1]]])
    elif M_rot[0, 0] != -1:
        axis = (1 / sympy.sqrt(2 * (1 + M_rot[0, 0]))) * np.array(
            [[1 + M_rot[0, 0]], [M_rot[1, 0]], [M_rot[2, 0]]])
    return axis

# Alias for backward compatibility
R_os_obrot = rotation_to_axis_angle

def inverse_homogeneous_transform(H: sympy.Matrix) -> sympy.Matrix:
    """
    Computes the inverse of a homogeneous transformation matrix.
    :param H: A 4x4 homogeneous transformation matrix.
    :return: The inverse of the homogeneous transformation matrix.
    """
    R = H[0:3, 0:3]
    t = H[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv * t
    H_inv = sympy.Matrix.zeros(4, 4)
    H_inv[0:3, 0:3] = R_inv
    H_inv[0:3, 3] = t_inv
    H_inv[3, 3] = 1
    return H_inv

# Alias for backward compatibility
inv_H = inverse_homogeneous_transform


def rotation_matrix(axis: list, theta: float) -> sympy.Matrix:
    """
    Creates a rotation matrix for a given axis and angle.
    :param axis: A list or array representing the axis of rotation.
    :param theta: The rotation angle in radians.
    :return: A 4x4 rotation matrix.
    """
    R = axis_angle_rotation(axis, theta)
    H = sympy.Matrix.zeros(4, 4)
    H[0:3, 0:3] = R
    H[3, 3] = 1
    return H
	
Rot = rotation_matrix



# Aliases for specific rotations
def Rx(theta: float) -> sympy.Matrix:
    """Rotation about the x-axis."""
    return rotation_matrix([1, 0, 0], theta)

def Ry(theta: float) -> sympy.Matrix:
    """Rotation about the y-axis."""
    return rotation_matrix([0, 1, 0], theta)

def Rz(theta: float) -> sympy.Matrix:
    """Rotation about the z-axis."""
    return rotation_matrix([0, 0, 1], theta)





def translation(translation_vector: list) -> sympy.Matrix:
    """
    Creates a translation matrix from a given vector.
    :param translation_vector: A 3-element list or array for translation.
    :return: A 4x4 translation matrix.
    """
    t = sympy.Matrix([translation_vector])
    H = sympy.Matrix.eye(4)
    H[0:3, 3] = t.T
    return H
    
Trans = trans = translation

# Aliases for specific translations
def Tx(tx: float) -> sympy.Matrix:
    """Translation along the x-axis."""
    return translation([tx, 0, 0])

def Ty(ty: float) -> sympy.Matrix:
    """Translation along the y-axis."""
    return translation([0, ty, 0])

def Tz(tz: float) -> sympy.Matrix:
    """Translation along the z-axis."""
    return translation([0, 0, tz])




def homogeneous_transform(axis: list, angle: float, t: list) -> sympy.Matrix:
    """
    Creates a homogeneous transformation matrix.
    :param axis: A list or array representing the axis of rotation.
    :param angle: The rotation angle in radians.
    :param t: A list or array representing the translation vector.
    :return: A 4x4 homogeneous transformation matrix.
    """
    return translation(t) * rotation_matrix(axis, angle)

# Alias for backward compatibility
H = homogeneous_transform
