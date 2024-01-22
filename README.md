# Closest Farthest Widest

The current paper proposes and tests algorithms for finding the diameter of a compact convex set $S$ and the farthest point $x$ in $S$ to a point $y$.
For these two nonconvex problems, we construct Frank-Wolfe and projected gradient ascent algorithms.
Although these algorithms are guaranteed to go uphill, they can get trapped by local maxima.
To avoid this defect, we investigate a homotopy method that gradually deforms a ball into the target set.
Motivated by the Frank-Wolfe algorithm, we also find the support function of the intersection of a convex cone and a ball centered at the origin and elaborate a known bisection algorithm for calculating the support function of a convex sublevel set.
The Frank-Wolfe and projected gradient algorithms are tested on five compact convex sets:

1. the box $[−1, 1]$,
2. the intersection of the unit ball and the non-negative orthant,
3. the probability simplex,
4. the $\ell_{1}$ unit ball, and
5. a sublevel set of the elastic net penalty.

Frank-Wolfe and projected gradient ascent are about equally fast on these test problem.
Ignoring homotopy, Frank-Wolfe algorithms is more reliable.
However, homotopy allows projected gradient ascent to recover from its failures.
