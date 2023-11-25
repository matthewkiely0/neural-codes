'''
This program is able to:

Compute the support of a neural code
Compute the simplicial complex of a neural code
Determine convexity up to:
    The code being a simplicial complex
    The code being intersection complete
    The code being max intersection complete
    The code containing any local obstructions to convexity (whether Cmin is contained in C)

    
Functions:

supp(C) - computes support of a code C
subsets(l) - computes all subsets of a list l
is_simplicial(C) - checks if the support of a code C is a simplicial complex
Delta(C) - computes the simplicial complex of a code C
Link(sigma, Delta) - computes the link of a simplicial complex Delta with respect to an element sigma in Delta

del_k(L,k) - computes the k-th boundary map of a link L

leftmult2 - These 3 functions work to compute the smith normal form of a matrix m (and other relevant objects)
rightmult2 - ^
smith(m) - ^ 

smith_rank(A) - Computes the rank of a matrix A by using the smith normal form (and uses a backup if this fails)
alpha_vector(L,k) - Outputs the non-zero diagonal elements of the k-th boundary map of the link L
alpha_matrix(L) - Outputs the non-zero diagonal elements of all (non-trivial*) boundary maps of the link L
Betti_Number(L,k) - Outputs the Betti-number of the k-th homology group of a link L
Betti_Vector(L) - Outputs the Betti-numbers of all (non-trivial*) homology groups of a link L

* We expect the homology groups H_k to consistently equal to 0 beyond a certain k. Hence the alpha's and betti-numbers become 'trivial'

C_min(C)
local_obstruction_exists(C)
intersection_complete(C)
supp_maximals(C)
maximals(C)
max_intersection_complete(C)
is_perm_equiv(C,D)

The investigation of local obstructions led to an interesting excursion into homology
'''
import numpy as np
from sympy import *
from itertools import combinations
from numpy.linalg import matrix_rank
import copy

# supp function
def supp(C):
    supp = []
    for vec in C:
        supp_c = []
        for index, neuron in enumerate(vec):
            if neuron == 1:
                supp_c.append(index + 1)    # +1 since want first neuron to be 1 not 0
        supp.append(supp_c)
    return supp

# code for generating subsets
def subsets(l):
    comb = []
    for i in range(len(l)+1):
        comb += [list(j) for j in combinations(l, i)]
    return comb

# testing if C is a simplicial complex
def is_simplicial(C):
    
    Supp_C = supp(C)
    
    for supp_c in Supp_C:
        subsets_c = subsets(supp_c)
        for element in subsets_c:
            if element not in Supp_C:
                return False
    return True

# construct minimal simplicial complex from a code
def Delta(C):
    
    Supp_C = supp(C)

    if is_simplicial(C):
        return sorted(Supp_C)

    missing_elements = []
    for supp_c in Supp_C:
        subsets_c = subsets(supp_c)
        for element in subsets_c:
            if element not in Supp_C:
                if element not in missing_elements:
                    missing_elements.append(element)

    S = sorted(Supp_C + missing_elements)
    
    if S != [[]]:
        if [] in S:
            S.remove([])
        
    if S == []:
        S.append([])

    return sorted(S)

# Construct a link
def Link(sigma, Delta):
    Link = []
    # sigma is a list
    # Delta is a list of lists
    if sigma not in Delta:
        print(f'The element: {sigma} must be in the simplicial complex: {Delta}.')

    for omega in Delta:
        # flag to check if there is a non-empty intersection between omega and sigma
        flag = 0
        for o in omega:
            if o in sigma:
                flag = 1

        if flag == 1:
            continue

        # omega does not intersect with sigma at this point
        if list(set(omega) | set(sigma)) in Delta:
            Link.append(omega)

    if Link != [[]]:
        if [] in Link:
            Link.remove([])
        
    if Link == []:
        Link.append([])

    return Link

# boundary map
def del_k(L,k):

    faces = [p for p in L if len(p) == k+1]

    if k == 0:
        array = [0 for f in faces]
        return Matrix(array).T

    if len(faces) == 0:
        return Matrix([0])

    edges = [p for p in L if len(p) == k]

    # iterate through faces for each edge
    array = []
    for edge in edges:
        row = []
        for face in faces:
            flag = 0
            for index, point in enumerate(face):
                face_remove = [p for p in face if p != point]
                if face_remove == edge:
                    if index % 2 == 0:
                        row.append(1)
                    if index % 2 == 1:
                        row.append(-1)
                    flag = 1
            if flag == 0:
                row.append(0)
        array.append(row)


    matrix = Matrix(array)
    return matrix

#3 Functions sourced directly from https://eric-bunch.github.io/blog/calculating_homology_of_simplicial_complex
#===================

def leftmult2(m, i0, i1, a, b, c, d):
    for j in range(m.cols):
        x, y = m[i0, j], m[i1, j]
        m[i0, j] = a * x + b * y
        m[i1, j] = c * x + d * y

def rightmult2(m, j0, j1, a, b, c, d):
    for i in range(m.rows):
        x, y = m[i, j0], m[i, j1]
        m[i, j0] = a * x + c * y
        m[i, j1] = b * x + d * y

def smith(m, domain=ZZ):
    try:
        m = Matrix(m)
        s = eye(m.rows)
        t = eye(m.cols)

        last_j = -1
        for i in range(m.rows):
            for j in range(last_j+1, m.cols):
                if not m.col(j).is_zero:
                    break
            else:
                break
            if m[i,j] == 0:
                for ii in range(m.rows):
                    if m[ii,j] != 0:
                        break
                leftmult2(m, i, ii, 0, 1, 1, 0)
                rightmult2(s, i, ii, 0, 1, 1, 0)
            rightmult2(m, j, i, 0, 1, 1, 0)
            leftmult2(t, j, i, 0, 1, 1, 0)
            j = i
            upd = True
            while upd:
                upd = False
                for ii in range(i+1, m.rows):
                    if m[ii, j] == 0:
                        continue
                    upd = True
                    if domain.rem(m[ii, j], m[i, j]) != 0:
                        coef1, coef2, g = domain.gcdex(m[i,j], m[ii, j])
                        coef3 = domain.quo(m[ii, j], g)
                        coef4 = domain.quo(m[i, j], g)
                        leftmult2(m, i, ii, coef1, coef2, -coef3, coef4)
                        rightmult2(s, i, ii, coef4, -coef2, coef3, coef1)
                    coef5 = domain.quo(m[ii, j], m[i, j])
                    leftmult2(m, i, ii, 1, 0, -coef5, 1)
                    rightmult2(s, i, ii, 1, 0, coef5, 1)
                for jj in range(j+1, m.cols):
                    if m[i, jj] == 0:
                        continue
                    upd = True
                    if domain.rem(m[i, jj], m[i, j]) != 0:
                        coef1, coef2, g = domain.gcdex(m[i,j], m[i, jj])
                        coef3 = domain.quo(m[i, jj], g)
                        coef4 = domain.quo(m[i, j], g)
                        rightmult2(m, j, jj, coef1, -coef3, coef2, coef4)
                        leftmult2(t, j, jj, coef4, coef3, -coef2, coef1)
                    coef5 = domain.quo(m[i, jj], m[i, j])
                    rightmult2(m, j, jj, 1, -coef5, 0, 1)
                    leftmult2(t, j, jj, 1, coef5, 0, 1)
            last_j = j
        for i1 in range(min(m.rows, m.cols)):
            for i0 in reversed(range(i1)):
                coef1, coef2, g = domain.gcdex(m[i0, i0], m[i1,i1])
                if g == 0:
                    continue
                coef3 = domain.quo(m[i1, i1], g)
                coef4 = domain.quo(m[i0, i0], g)
                leftmult2(m, i0, i1, 1, coef2, coef3, coef2*coef3-1)
                rightmult2(s, i0, i1, 1-coef2*coef3, coef2, coef3, -1)
                rightmult2(m, i0, i1, coef1, 1-coef1*coef4, 1, -coef4)
                leftmult2(t, i0, i1, coef4, 1-coef1*coef4, 1, -coef1)
        return (s, m, t, 0)
    except (ValueError, ZeroDivisionError):
        return (0, m, 0, 1)
#===================
'''
The rank of an integer matrix A can be determined by finding the smith form of A = QDP, and finding the rank of D.
The rank of D can be determined by counting the number of non-zero elements along the diagonal 
'''
def smith_rank(A):

    S = smith(A)

    # if smith works
    if S[3] == 0:
        D = S[1]

        rows = shape(D)[0]
        cols = shape(D)[1]

        rank = 0
        for i in range(min([rows,cols])):
            if (D.row(i)).col(i) != Matrix([[0]]):
                rank = rank + 1

        return rank

    # if it doesnt
    else:
        m = np.array(A).astype(np.float64)
        rank = matrix_rank(m)
        return rank
    
'''
To prove contractibility, we expect the set of Betti-numbers for a set to be [1,0,0,0,...]
and the alpha matrix to be all zeroes or ones
'''

# obtains the diagonal elements of the matrix D mentioned in the smith normal form for del_k
def alpha_vector(L, k):

    delk1 = del_k(L,k+1)
    S = smith(delk1)
 
    if S[3] == 0:
        D = S[1]

        rows = shape(D)[0]
        cols = shape(D)[1]

        minsize = min(rows, cols)

        alphas = []

        for i in range(minsize):
            alphas.append((D.row(i).col(i)).det()) #really ugly way to extract value

        return alphas
    
# obtains the alpha_vectors for all k
def alpha_matrix(L):
    Mat = []
    sizes = [len(l) for l in L]
    n = max(sizes)

    for k in range(n):
        Mat.append(alpha_vector(L,k))

    return Mat

# obtains the betti number for the kth homology group
def Betti_Number(L,k):

    delk = del_k(L,k)
    delk1 = del_k(L,k+1) 

    m = shape(delk)[1]

    s = smith_rank(delk)
    r = smith_rank(delk1)

    return m - r - s

# obtains the betti numbers for all homology groups
def Betti_Vector(L):
    
    if L:
        vec = []
        sizes = [len(l) for l in L]
        n = max(sizes)

        for k in range(n):
            vec.append(Betti_Number(L,k))

    else:
        return []

    return vec

# computes Cmin
def C_min(C):
    
    supp_C_min = [[]] # c_min = {...} + \emptyset

    delta = Delta(C)

    for sigma in delta:
        
        link = Link(sigma, delta)
        alpha = alpha_matrix(link)
        betti = Betti_Vector(link)

        flag = 0
        if betti:
            for index, b in enumerate(betti):
                if index == 0:
                    if b != 1:
                        flag = 1
                else:
                    if b != 0:
                        flag = 1

        for K in alpha:
            if K:
                for i in K:
                    if i not in [0,1,-1]:
                        flag = 1
        if flag == 1:
            if sigma == []:
                pass
            else:
                supp_C_min.append(sigma)

    return supp_C_min

# checks whether Cmin is a subset of C, and hence whether C can be determined to be non-convex
def local_obstruction_exists(C):

    Cm = C_min(C)

    flag = 0
    for cm in Cm:
        if cm != []:
            flag = 1
    if flag == 0:
        return False

    for cm in Cm:
        if cm != []:
            if cm not in supp(C):
                return True

    return False
        
# checks if a code is intersection complete, and hence convex
def intersection_complete(C):
    for c in supp(C):
        for d in supp(C):
            intersection = [i for i in c if i in d]
            
            if c == []:
                intersection = d

            if d == []:
                intersection = c

            if c == [] and d == []:
                intersection = []

            if intersection not in supp(C):
                return False
    return True

# computes the maximal elements of suppC
def supp_maximals(supp_C):
    D = []
    # if d is not a subset of anything excluding itself it is a maximal set
    for d in supp_C:
        flag = 0
        for g in supp_C:
            if set(d).issubset(set(g)) and g != d:
                flag = 1
        if flag == 1:
            continue
        else:
            D.append(d)
    return sorted(D)

# computes the maximal codewords of C
def maximals(C):

    if C:
        n = max([len(c) for c in C])

    Supp_C = supp(C)
    maximals = supp_maximals(Supp_C)

    maximals_C = []
    for M in maximals:
        c = [0 for i in range(n)]
        
        for m in M:
            c[m-1] = 1
        maximals_C.append(c)

    return maximals_C

# checks if a code is max-intersection complete, and hence convex
def max_intersection_complete(C):
    
    facets = supp_maximals(Delta(C))

    # we want to search the subsets of facets
    L = len(facets)
    indices = list(range(1, L+1))

    combos = []
    for i in range(1,L+1):
        for element in combinations(indices, i):
            combos.append(list(element))
            
    for combo in combos:
        X = []
        subset = []
        for com in combo:
            subset.append(facets[com-1])

        if len(subset) == 1:
            X.append(subset[0])
            X = X[0]

        if len(subset) > 1:

            x = set(subset[0])
            for y in subset[1:]:
                x = x.intersection(set(y))
            X.append(list(x))
            X = X[0]

        if X not in supp(C):
            return False

    return True

# checks if two codes are equivalent with respect to swapping neuron labels
def is_perm_equiv(C,D):


    # if both codes are empty they are perm equiv
    if not C and not D:
        return True

    # if only one code is empty they are not perm equiv
    if bool(C) != bool(D):
        return False

    # only case left is that they are both non-empty
    L = max([len(c) for c in C])
    M = max([len(d) for d in D])

        
    if L != M:
        return False
    
    if [] in C:
        C.remove([])
    if [] in D:
        D.remove([])

    Pairs = [(0,0)]
    for i in range(L):
        for j in range(i):  
            Pairs.append((i,j))

    PEs = []
    for (i,j) in Pairs:
        E = copy.deepcopy(D)
        for e in E:
            e[i],e[j] = e[j],e[i]
        PEs.append(E)

    for P in sorted(PEs):
        if sorted(C) == sorted(P):
            return True
    return False

# checks all discussed conditions of convexity/non-convexity on a code C
def convexity(C):

    if C:
        l = [len(c) for c in C]
        L = max(l)

        ones = [1 for i in range(L)]
        zeroes = [0 for i in range(L)]

        if ones in C:
            return True, 'All ones codeword'

    if supp(C) == Delta(C):
        return True, 'Simplicial Complex'
    
    elif intersection_complete(C):
        return True, 'Intersection-Complete'

    # not fully tested! (and not correct!)
    elif max_intersection_complete(C):
        return True, 'Max-Intersection-Complete'

    elif zeroes in C:
        if local_obstruction_exists(C):
            return False, 'Local Obstruction'

    else:
        return None, 'Cannot determine convexity!'