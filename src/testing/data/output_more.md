# Conversely, if q is also particular solution, then p ▯ q is a solution to the homogeneous system, since

A(p ▯ q) = Ap ▯ Aq = b ▯ b = 0.

Thus, p ▯ q is an element of N (A). Therefore q = p + x, where x = q ▯ p, as asserted. This completes the proof.

In the above proof, we made the statement that A(p + x) = Ap + Ax. This follows from a general algebraic identity called the distributive law which we haven’t yet discussed. However, our particular use of the distributive law is easy to verify from first principles.

Example 2.8. Consider the system involving the counting matrix C of Example 2.5:

1x1 + 2x2 + 3x3 = a
4x1 + 5x2 + 6x3 = b
7x1 + 8x2 + 9x3 = c

where a, b and c are fixed arbitrary constants. This system has augmented coefficient matrix

0 1 2 3 a
4 5 6 b
7 8 9 c

We can use the same sequence of row operations as in Example 2.5 to put (C|b) into reduced form (C red|c) but to minimize the arithmetic with denominators, we will actually use a different sequence.

1 2 3 a
3 3 3 b ▯ a
7 8 9 c
1 2 3 a
1 2 3 a
3 3 3 b ▯ a
1 2 3 c ▯ 2b + 2a
1 2 3 a
3 3 3 b ▯ a
1 2 3 a
1 2 3 a
3 3 3 b ▯ a
1 2 3 c ▯ 2b + a
1 2 3 a
3 3 3 b ▯ a
1 2 3 a
1 0 ▯1 (▯5/3)a + (2/3)b
0 1 2 (4/3)a ▯ (1/3)b
0 0 0 c ▯ 2b + a