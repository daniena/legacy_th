from numpy import *
from numpy.linalg import *
from param_debug import debug

def vector(v):
    return transpose(array([v]))

def dagger(M):
    #return dot(transpose(M),inv(dot(M,transpose(M))))
    return pinv(M)

def multidot(Mlist):
    l = len(Mlist) - 1
    M = dot(Mlist[l-1],Mlist[l])

    if debug:
        print('l:', l)
        print('M first dot:', M)

    for m in range(l-1,0,-1):
        M = dot(Mlist[m-1],M)
        if debug:
            print('M dot nr', m, ':', M )
    return M

def double_equality(a, b, precision):
    if a + precision >= b and a - precision <= b:
        return True
    else:
        if debug:
            print('double_equality: a:', a, 'b:', b)
        return False

def matrix_double_equality(matrix_a, matrix_b, precision):
    if matrix_a.shape != matrix_b.shape:
        return False
    
    N = matrix_a.shape[0]
    M = matrix_a.shape[1]
    
    if debug:
        print('N:', N, 'M:', M)

    for i in range(0,N):
        for j in range(0,M):
            if not double_equality(double(matrix_a[i][j]), double(matrix_b[i][j]), precision):
                if debug:
                    print('i:', i, 'j:', j)
                    print('matrix_a[i][j]:', matrix_a[i][j], 'matrix_b[i][j]:', matrix_b[i][j])
                return False
    return True
