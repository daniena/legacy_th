from numpy import *
from util.matrix import *
from util.rotation import *
from util.file_operations import *
from param_debug import debug
import os

def test_matrix_double_equality():
    print('')
    # Tests whether matrix_double_comparator works
    
    matrix_A = array([[0.2682, 0.2124, 0.1702, 0.0927, -0.0784, -0.0000], [-0.7639, 0.0213, 0.0171, 0.0093, 0.0004, 0.0000], [0, -0.7868, -0.3639, 0.0203, -0.0241, -0.0000], [0, 0.0998, 0.0998, 0.0998, 0.2940, 0.0044], [0, -0.9950, -0.9950, -0.9950,  0.0295, -0.9996], [1.0000, 0, 0, 0, -0.9553, -0.0295]])
    matrix_B = array([[0.2672, 0.2124, 0.1702, 0.0927, -0.0784, -0.0000], [-0.7639, 0.0213, 0.0171, 0.0093, 0.0004, 0.0000], [0, -0.7868, -0.3639, 0.0203, -0.0241, -0.0000], [0, 0.0998, 0.0998, 0.0998, 0.2940, 0.0044], [0, -0.9950, -0.9950, -0.9950,  0.0295, -0.9996], [1.0000, 0, 0, 0, -0.9553, -0.0295]])
    matrix_C = array([[0.26725, 0.2124, 0.1702, 0.0927, -0.0784, -0.0000], [-0.7639, 0.0213, 0.0171, 0.0093, 0.0004, 0.0000], [0, -0.7868, -0.3639, 0.0203, -0.0241, -0.0000], [0, 0.0998, 0.0998, 0.0998, 0.2940, 0.0044], [0, -0.9950, -0.9950, -0.9950,  0.0295, -0.9996], [1.0000, 0, 0, 0, -0.9553, -0.0295]])
    matrix_D = array([[0.26705, 0.2124, 0.1702, 0.0927, -0.0784, -0.0000], [-0.7639, 0.0213, 0.0171, 0.0093, 0.0004, 0.0000], [0, -0.7868, -0.3639, 0.0203, -0.0241, -0.0000], [0, 0.0998, 0.0998, 0.0998, 0.2940, 0.0044], [0, -0.9950, -0.9950, -0.9950,  0.0295, -0.9996], [1.0000, 0, 0, 0, -0.9553, -0.0295]])
    
    if debug:
        print('_____test_matrix_double_equality_____')
        print('matrix_A')
        print(matrix_A)
        print('matrix_B')
        print(matrix_B)
        print('matrix_C')
        print(matrix_C)
        print('matrix_D')
        print(matrix_D)
    
    #Single element tests:
    # Test 1:
    # Test whether equality works:
    print('Test: matrix_double_comparator_1 equality, ASSERT TRUE:', matrix_double_equality(matrix_A,matrix_A,0.0001))

    # Test 2
    # Test whether inequality works:
    print('Test: matrix_double_comparator_2a inequality, ASSERT TRUE:', not matrix_double_equality(matrix_B,matrix_A,0.0001))
    print('Test: matrix_double_comparator_2b inequality, ASSERT TRUE: ', not matrix_double_equality(matrix_A,matrix_B,0.0001))

    # Test 3
    # Test whether inequality fails within border of precision by returning a false positive:
    print('Test: matrix_double_comparator_3 imprecise false positive, ASSERT TRUE:', matrix_double_equality(matrix_B,matrix_C,0.0001))

    # Test 4
    # Test whether inequality succeeds just outside of border of precision by returning false:
    print('Test: matrix_double_comparator_4 imprecise true negative, ASSERT TRUE:', not matrix_double_equality(matrix_B,matrix_D,0.0001))

    #Several elements tests:
    # Only if necessary...
    print('')

def test_multidot():
    print('')

    a = diag([1, 1])
    b = array([[1, 1], [1, 1]])
    c = array([[1, 2, 3], [1, 2, 3]])
    d = array([[0, 1], [0, 1], [1, 0]])

    result1 = multidot([a, b, c])
    correct1 = array([[2, 4, 6], [2, 4, 6]])

    result2 = multidot([a, b, c, d])
    correct2 = array([[6, 6], [6, 6]])

    if debug:
        print('_____test_multidot_____')
        print('a')
        print(a)
        print('b')
        print(b)
        print('c')
        print(c)
        print('d')
        print(d)

        print('result1')
        print(result1)
        print('correct1')
        print(correct1)

        print('result2')
        print(result2)
        print('correct2')
        print(correct2)

    # Test 1:
    print('Test: multidot_1a yields correct result, ASSERT TRUE:', matrix_double_equality(result1, correct1, 0.001))
    print('Test: multidot_1b yields correct result, ASSERT TRUE:', matrix_double_equality(result2, correct2, 0.001))

    #Only write more tests if discovered necessary

    print('')

def test_dagger():
    print('')

    J = array([[1, 1, 1, 1, 0, 1], [1, 2, 1, 1, 2, 1], [1, 1, 1, 1, 2, 1]])
    result = dagger(J)
    correct = array([[0.25, -0.25, 0.25], [0, 1, -1], [0.25, -0.25, 0.25], [0.25, -0.25, 0.25], [-0.5, 0, 0.5], [0.25, -0.25, 0.25]])
    
    if debug:
        print('_____test_dagger_____')
        print('J')
        print(J)
        print('result')
        print(result)
        print('correct')
        print(correct)

    # Test 1:
    print('Test: dagger yields correct result, ASSERT TRUE:', matrix_double_equality(result, correct, 0.001))

    #Only write more tests if discovered necessary

    print('')

def test_axis_angle_rotation_matrix():
    print('')
    
    ks = (vector([0,0,1]), vector([0,1,0]), vector([1,0,0]), vector([0,0,1]))
    thetas = (0.1, 0.1, 0.1, -pi/4)


    for k, theta in zip(ks, thetas):
        R = axis_angle_rotation_matrix(k, theta)
        assert is_rotation_matrix(R)

    print('Test: axis_angle_rotation_matrix, ASSERT TRUE:', True)
    print('')
    

def test_remove_files_with_ending():
    print('')
    
    path = os.getcwd() + '/test/testdir'
    os.mkdir(path)
    with open(path + '/testfile.txt', 'w') as testfile:
        testfile.write('test')
    if debug:
        print('os.listdir(path):')
        print(os.listdir(path))
    assert os.listdir(path)
    
    remove_files_with_ending(path, '.txt')
    if debug:
        print('os.listdir(path):')
        print(os.listdir(path))
    assert not os.listdir(path)

    with open(path + '/testfile1.txt', 'w') as testfile:
        testfile.write('test1')
    with open(path + '/testfile2.txt', 'w') as testfile:
        testfile.write('test2')
    if debug:
        print('os.listdir(path):')
        print(os.listdir(path))
    assert len(os.listdir(path)) == 2

    remove_files_with_ending(path, '.txt')
    if debug:
        print('os.listdir(path):')
        print(os.listdir(path))
    assert not os.listdir(path)

    os.rmdir(path)

    print('Test: remove_files_with_ending removes all files, ASSERT TRUE:', True)
    print('')
    
def test_make_path():
    print('')

    path = os.getcwd() + '/test/testdir'
    assert not os.path.isdir(path)

    make_path(path)
    assert os.path.isdir(path)

    os.rmdir(path)

    deeper_path = path + '/testsubdir'
    assert not os.path.isdir(path)

    make_path(deeper_path)
    assert os.path.isdir(path)
    assert os.path.isdir(deeper_path)

    os.rmdir(deeper_path)
    os.rmdir(path)

    print('Test: make_path generates paths correctly, ASSERT TRUE:', True)
    print('')
