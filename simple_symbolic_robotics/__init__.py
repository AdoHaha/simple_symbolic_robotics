#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy
from numpy import cross, eye, dot
from scipy.linalg import norm
import scipy
import sympy

from sympy import lambdify, pprint, pi

'''Python package providing some simple symbolic helper functions useful in
kinematics calculations'''


def S_sym(wektor):
    '''returns skew symmetric matrix for wector wektor'''

    S = sympy.Matrix([[0, -wektor[2], wektor[1]], 
    [wektor[2], 0, -wektor[0]], [-wektor[1], wektor[0], 0]])

    return S



def is_rotation_matrix(matrix):
    w1 = matrix*matrix.T == matrix.T*matrix == diag(1,1,1)
    return w1

def os_obrot(wek, kat, p_flag=False):
    '''funkcja zwraca postać macierzy R przy obrocie wokół wektora w
    o dany kąt, w formie jak najbardziej symboliczne'''

    w = sympy.Matrix(wek)
    w = w / w.norm()

    A = S_sym(w)

    A_kw = A * A

    wynik = sympy.eye(3) + sympy.sin(kat) * A + (1 - sympy.cos(kat)) * A_kw
    if(p_flag):
        print("wektor", wek)
        print("znormalizowany wektor \n", w)
        print("Skosnie symetryczna ze znorm \n", A)
        print("Kwadrat ze skosniesymetrycznej \n", A.dot(A))
        print("wynik \n", wynik)
    #print("wynik \n",wynik)
    return wynik


def R_os_obrot(R):
    '''oblicza oś i kąt z macierzy R, algorytm zaczerpnięty z kursu SNUx na edx
    returns theta and axis'''
    M_rot = numpy.array(R)
    tr = numpy.trace(M_rot)
    if(numpy.array_equal(M_rot, eye(3))):
        theta = 0
        w = numpy.zeros((3, 1))  # nothing
    elif(tr == -1):
        theta = sympy.pi
        if(M_rot[2, 2] != -1):
            w = (1 / sympy.sqrt(2 * (1 + M_rot[2, 2]))) * numpy.array(
                [[M_rot[0, 2]], [M_rot[1, 2]], [1 + M_rot[2, 2]]])
        elif(M_rot[1, 1] != -1):
            w = (1 / sympy.sqrt(2 * (1 + M_rot[1, 1]))) * numpy.array(
                [[M_rot[0, 1]], [1 + M_rot[1, 1]], [M_rot[2, 1]]])
        elif(M_rot[0, 0] != -1):
            w = (1 / sympy.sqrt(2 * (1 + M_rot[0, 0]))) * numpy.array(
                [[1 + M_rot[0, 0]], [M_rot[1, 0]], [M_rot[2, 0]]])
    else:
        theta = acos((tr - 1) / 2)
        print(theta)
        MM = M_rot - M_rot.T
        w = (1 / (2 * sin(theta))) * \
            sympy.Matrix([[MM[2, 1]], [MM[0, 2]], [MM[1, 0]]])
    return w, theta

def inv_H(H):
    '''returns inverse of H'''
    R = H[0:3, 0:3]
    t = H[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv * t
    H_inv = sympy.Matrix.zeros(4, 4)
    H_inv[0:3, 0:3] = R_inv
    H_inv[0:3, 3] = t_inv
    H_inv[3, 3] = 1
    return H_inv
	

def Rot(wek, theta):
    R = os_obrot(wek, theta)
    H = sympy.Matrix.zeros(4, 4)
    H[0:3, 0:3] = R
    H[3, 3] = 1
    return H


def Rx(theta):
    H = Rot([1, 0, 0], theta)
    return H


def Ry(theta):
    H = Rot([0, 1, 0], theta)
    return H


def Rz(theta):
    H = Rot([0, 0, 1], theta)
    return H



def trans(wek):
    wek = sympy.Matrix([wek])
    H = sympy.Matrix.eye(4)
    H[0, 3] = wek[0]
    H[1, 3] = wek[1]
    H[2, 3] = wek[2]
    return H
    
Trans = trans

def Tx(tx):
    return trans([tx,0,0])

def Ty(ty):
    return trans([0,ty,0])
def Tz(tz):
    return trans([0,0,tz])

def H(axis,angle,t):
    return trans(t)*Rot(axis, angle)
