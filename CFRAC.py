import math
import numpy as np
import copy
from decimal import *
import time
import sympy

getcontext().prec = 50

def find_primes(upper_bound):   #Finds all prime op to upper_bound
    List = []
    for i in range(2,upper_bound):
        for j in range(2,i):
            if (i % j) == 0:
                break
        else:
            List.append(i)
    return List

def jacobi_symbol(a, N):    #Find the Jacobi symbol of a mod N (equals Lagrange symbol when N prime)
    value = 1
    while True:
        if a > N:
            a = a % N
        if a == 0:
            return 0
        while a % 2 == 0:
            a = a//2
            if N % 8 == 3 or N % 8 == 5:
                value *= -1
        if a == 1:
            return value
        if math.gcd(a, N) != 1:
            return 0
        if a % 4 == 3 and N % 4 == 3:
            value *= -1
        a, N = N, a

def find_quadratic_primes(N, prime_bound):  #Finds all primes p, where N is quadratic mod p up to an upper bound
    prime_list = sympy.primerange(1,prime_bound)
    quadratics = []
    
    for prime in prime_list:
        jaco = jacobi_symbol(N, prime)
        if jaco == 1 or jaco == 0:
            quadratics.append(prime)
    return quadratics

def trial_division_early_stopping(n, prime_list, prime_bound, expand_base, N):    #Factorizes numbers by a given factor base by trial division
    sqrtN = Decimal(N).sqrt()
    factors_list = []
    new_prime = 0
    if n < 0:
        a = -n
        factors_list.append(-1)
    else:
        a = n
    for i in prime_list[0:15]:        
        while (a % i) == 0:
            factors_list.append(i)
            a = a//i
    if a > sqrtN//500:
        return 'Early', 'Abort'
    for i in prime_list[15:95]:        
        while (a % i) == 0:
            factors_list.append(i)
            a = a//i
    if a > sqrtN//(2*(10**7)):
        return 'Early', 'Abort'
    for i in prime_list[95:]:        
        while (a % i) == 0:
            factors_list.append(i)
            a = a//i
    if expand_base == True:
        if 1 < a < prime_bound**2:
            factors_list.append(a)
            new_prime = a
            return factors_list, new_prime               
    if a != 1:
        return 'No factorization of small enough primes', 'No'
        
    return factors_list, new_prime

def calculate_terms(N, n):  #Calculates Q_n, P_n and q_n op to given bound for n
    sqrtN = Decimal(N).sqrt()
    Q_list = [int(1)]
    P_list = [0]
    q_list = [math.floor(sqrtN)]

    for i in range(1,n):
        P_list.append(q_list[i-1]*Q_list[i-1] - P_list[i-1])
        Q_list.append((N - P_list[i]**2)//Q_list[i-1])
        q_list.append(math.floor((sqrtN + P_list[i])//Q_list[i]))
        
    return P_list, Q_list, q_list

def calculate_Ai(q_list, n, N): #Calculates A_i
    A_list = [q_list[0], (q_list[0]*q_list[1] + 1)%N]
    
    for i in range(2,n):
        A_list.append((q_list[i]*A_list[i-1]+A_list[i-2])%N)
        
    return A_list

def factorize_Qi(Q_list, factor_base, prime_bound, expand_base, N): #Factorizes Q_n from a list with a given factor_base, and expands the base if expand_base=True
    factor_base_copy = copy.deepcopy(factor_base)
    factors_list = []
    Q_index = []
    new_primes = []
    Q_length = len(Q_list)
    for i in range(1, Q_length):
        if i%100000 == 0:
            print('Factorizing %d of %d' %(i, Q_length))
        factors, new_prime = trial_division_early_stopping(Q_list[i], factor_base_copy, prime_bound, expand_base, N)
        if isinstance(factors, list):
            factors_list.append(factors)
            Q_index.append(i)
            new_primes.append(new_prime)
    new_primes = list(set(new_primes))
    return factors_list, Q_index, new_primes

def convert_factors_to_binary(factorizations_list, prime_list): #Converts list of lists of factors to binary matrix
    prime_length = len(prime_list)
    prime_index = {}
    matrix = []

    for i in range(prime_length):
        prime_index[prime_list[i]] = i

    for fac in factorizations_list:
        L = [0]*prime_length
        for j in fac:
            index = prime_index[j]
            L[index] = (L[index] + 1)%2
        matrix.append(L)

    return matrix

def gauss_elimination(matrix): #Does Gauss elimination on binary matrix and history matrix
    matrix = np.array(matrix)
    column_size = len(matrix[0])
    row_size = len(matrix)
    identity = np.identity(row_size, dtype = int)

    for column in range(column_size-1, -1, -1):
        if column%1000 == 0:
            print('Checking column %d of %d' %(column, column_size))
        pivot = False
        for row in range(row_size):
            if matrix[row][column] == 1 and np.sum(matrix[row][column+1:]) == 0:
                if not pivot:
                    pivot_row = matrix[row]
                    identity_row = identity[row]
                    pivot = True
                else:
                    matrix[row] = np.mod(np.add(matrix[row], pivot_row),2)
                    identity[row] = np.mod(np.add(identity[row], identity_row),2)
    return matrix, identity

def matrix_to_index(matrix, index_matrix, Q_index, prime_list): #Finds 0-vectors and converts them back to indexes
    matrix = np.array(matrix)
    sum_matrix = np.sum(matrix, axis=1)
    index_list = []
    
    for i in range(len(sum_matrix)):
        if sum_matrix[i] == 0:
            index, = np.where(index_matrix[i]==1)
            index_list.append([Q_index[j] for j in index])
    return index_list

def test_for_equivalence(N, Q_list, index_list, A_list):    #Checks for non-trivial factors
    non_equivalent = []
    for indexes in index_list:
        Q = int(Q_list[indexes[0]])
        sqrtR = 1
        A_product = A_list[indexes[0]-1]
        for index in indexes[1:]:
            value = int(Q_list[index])
            GCD = math.gcd(Q,value)
            Q = (Q//GCD)*(value//GCD)
            sqrtR *= GCD
            A_product = (A_product*A_list[index-1])%N
        root =  (sqrtR*int(Decimal(Q).sqrt()))%N
        A_product = A_product % N
        if (A_product + root)%N != 0 and (A_product - root)%N != 0:
            non_equivalent.append(math.gcd(N, A_product + root))
            non_equivalent.append(math.gcd(N, A_product - root))
    return non_equivalent

def find_factors(N, prime_bound, Q_bound, k, expand_base=True): #Compiles all functions above
    start = time.time()
    prime_bound += 1
    print('Finding quadratic primes...')
    factor_base = find_quadratic_primes(k*N, prime_bound)
    print('Elements in factor base:', len(factor_base))
    print('Calculating Q_n...')
    P_list, Q_list, q_list = calculate_terms(k*N, Q_bound)
    for i in range(len(Q_list)):
        if i % 2:
            Q_list[i] = Q_list[i]*(-1)
    print('Calculating A_n...')
    Ai_list = calculate_Ai(q_list, Q_bound, N)
    print('Factorizing Q_n...')
    start_factor = time.time()
    Q_factors, Q_index, new_primes = factorize_Qi(Q_list, factor_base, prime_bound, expand_base, N)
    end_factor = time.time()
    print('Time factoring:', end_factor - start_factor)
    print('Number of Q_n: ', len(Q_factors))
    if expand_base == True:
        factor_base = factor_base + new_primes
    print('Converting to binary matrix...')
    binary = convert_factors_to_binary(Q_factors, [-1] + factor_base)
    print('Gaussian elimination...')
    reduced, indexes = gauss_elimination(binary)
    print('Converting matrix to indexes...')
    index_list = matrix_to_index(reduced, indexes, Q_index, factor_base)
    print('Number of zero-vectors:', len(index_list))
    print('Finding factors...')
    equivalences = test_for_equivalence(N, Q_list, index_list, Ai_list)
    factors = list(set(equivalences))
    end = time.time()
    print('Time elapsed: ', end-start)
    if factors == []:
        print('No factors found :(')
    else:
        print('Factors found are: ', factors)
    
    return factors


find_factors(N = 2**128 + 1, prime_bound = 52183, Q_bound = 1293846, k = 257, expand_base=False)



