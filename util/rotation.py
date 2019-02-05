from math import *
from numpy import *
from util.matrix import *
from param_debug import debug

def is_rotation_matrix(M):
    det = linalg.det(M)

    inverse_equal_to_its_transpose = matrix_double_equality(linalg.inv(M), transpose(M), 0.0001)
    determinant_is_one_or_minus_one = double_equality(det, 1, 0.001) is True or double_equality(det, -1, 0.0001) is True

    if debug:
        print('inverse_equal_to_its_transpose')
        print(inverse_equal_to_its_transpose)
        print('determinant_is_one_or_minus_one')
        print(determinant_is_one_or_minus_one)
        print('det')
        print(det)
        print('M')
        print(M)
        print('linalg.inv(M)')
        print(linalg.inv(M))
        print('transpose(M)')
        print(transpose(M))
    if inverse_equal_to_its_transpose and determinant_is_one_or_minus_one:
        return True
    else:
        return False
        

def axis_angle_rotation_matrix(k, theta):
    kx = asscalar(k[0])
    ky = asscalar(k[1])
    kz = asscalar(k[2])

    t = theta    

    def c(theta):
        return cos(theta)
    def s(theta):
        return sin(theta)
    def v(theta):
        return 1 - cos(theta)
    
    R = array([[kx*kx*v(t) + c(t)    , kx*ky*v(t) - kz*s(t) , kx*kz*v(t) + ky*s(t)],
               [kx*ky*v(t) + kz*s(t) , ky*ky*v(t) + c(t)    , ky*kz*v(t) - kx*s(t)],
               [kx*kz*v(t)-ky*s(t)   , ky*kz*v(t) + kx*s(t) , kz*kz*v(t) + c(t)   ]])
    
    return R

def quat2rot():
    pass

def rot2quat():
    pass
