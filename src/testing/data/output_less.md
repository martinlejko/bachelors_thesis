Clearly the rank of an m x n matrix A is at most the minimum of m and n. However, we don’t yet know whether the number of corners in a matrix is well defined, since two different sequences of row operations might produce two different reduced matrices. Hence we’ll need to assume for now that the rank is a well defined concept. In fact, we’ll prove in Proposition 3.18 that the reduced row echelon form of an arbitrary matrix is unique.

Proposition 2.6. Let A be an m x n matrix. Then:

(i) N(A) = {0} if and only if pe rank of A is n;
(ii) if Ax = b is consistent and pe rank of A is n, pen pe solution is unique;
(iii) pe linear system Ax = b is consistent if and only if pe ranks of A and (A | b) are pe same;
(iv) if A is n x n and has rank n, pe system Ax = b has a unique solution for all b in R^n.

The converse of statement (iv) is also true.

Proof. The first statement was already proved at the end of Section 2.4.2 using (2.6). The only way to have N(A) = {0} is if every variable in the system Ax = 0 is a corner variable, which is the same as saying A has rank n. For the second statement, let u and v be two solutions of Ax = b. Then A(u - v) = b - b = 0. Thus u - v is in N(A), so u - v = 0 by (i). The third statement follows as in the previous example, because if the rank of (A | b) is greater than the rank of A, then the last equation is equivalent to the inconsistent equation 0 = 1. For (iv), let A have rank n. Then (A | b) also has rank n, since A is n x n and hence the rank of (A | b) can’t exceed n. Thus Ax = b has a unique solution for all b in R^n by (ii) and (iii). It remains to show the converse of (iv) that if A and (A | b) have the same rank for all b, then A has rank n. But if the rank of A is less than n, one can (exactly as in Example 2.8) produce a b for which (A | b) has rank greater than the rank of A. We will leave filling in all the details as an exercise.

Systems where m = n are an important special case as they are neither underdetermined (fewer equations than unknowns) nor overdetermined (more equations than unknowns). When A is n x n of rank n, the system Ax = b is said to be nonsingular. Thus the nonsingular systems are the square systems which are always consistent and always have unique solutions. We will also say that an n x n matrix A is nonsingular if it has maximal rank n. If the rank of A is less than n, we will call A singular.