# Parallel-Programming_4-1-Blocked-All-Pairs-Shortest-Path

## Goal

This assignment helps you get familiar with CUDA by implementing a blocked all-pairs shortest path algorithm. Besides, to measure the performance and scalability of your program, experiments are required. Finally, we encourage you to optimize your program by exploring different optimizing strategies for optimization points.


## Blocked Floyd-Warshall algorithm

Given an V×V
matrix W=[w(i,j)] where w(i,j)≥0 represents the distance (weight of the edge) from a vertex i to a vertex j in a directed graph with V vertices. We define an V×V matrix D=[d(i,j)] where d(i,j) denotes the shortest-path distance from a vertex i to a vertex j. Let D(k)=[d(k)(i,j)] be the result which all the intermediate vertices are in the set {0,1,2,...,k−1}

.

We define d(k)(i,j)

as the following:
d(k)(i,j)={w(i,j)min(d(k−1)(i,j),d(k−1)(i,k−1)+d(k−1)(k−1,j))if k=0;if k≥1.

The matrix D(V)=d(V)(i,j)

gives the answer to the all-pairs shortest path problem.

In the blocked all-pairs shortest path algorithm, we partition D
into ⌈V/B⌉×⌈V/B⌉ blocks of B×B submatrices. The number B is called the blocking factor. For instance, in figure 1, we divide a 6×6 matrix into 3×3 submatrices (or blocks) by B=2

.
![image](https://user-images.githubusercontent.com/36917138/139832648-e693336d-1e4b-46ec-80df-efd9bab015ba.png)

The blocked version of the Floyd-Warshall algorithm will perform ⌈V/B⌉ rounds, and each round is divided into 3 phases. It performs B

iterations in each phase.

Assuming a block is identified by its index (I,J)
, where 0≤I,J<⌈V/B⌉. The block with index (I,J) is denoted by D(k)(I,J)

.

In the following explanation, we assume N=6
and B=2

. The execution flow is described step by step as follows:

    Phase 1: self-dependent blocks.

    In the k

-th round, the first phase is to compute the B×B pivot block D(k⋅B)(k−1,k−1)

.

For instance, in the 1st round, D(2)(0,0)

is computed as follows:
d(1)(0,0)=min(d(0)(0,0),d(0)(0,0)+d(0)(0,0))d(1)(0,1)=min(d(0)(0,1),d(0)(0,0)+d(0)(0,1))d(1)(1,0)=min(d(0)(1,0),d(0)(1,0)+d(0)(0,0))d(1)(1,1)=min(d(0)(1,1),d(0)(1,0)+d(0)(0,1))d(2)(0,0)=min(d(1)(0,0),d(1)(0,1)+d(1)(1,0))d(2)(0,1)=min(d(1)(0,1),d(1)(0,1)+d(1)(1,1))d(2)(1,0)=min(d(1)(1,0),d(1)(1,1)+d(1)(1,0))d(2)(1,1)=min(d(1)(1,1),d(1)(1,1)+d(1)(1,1))

Note that the result of d(2)
depends on the result of d(1) and therefore cannot be computed in parallel with the computation of d(1)

.

Phase 2: pivot-row and pivot-column blocks.

In the k
-th round, it computes all D(k⋅B)(h,k−1) and D(k⋅B)(k−1,h) where h≠k−1

.

The result of pivot-row / pivot-column blocks depend on the result in phase 1 and itself.

For instance, in the 1st round, the result of D(2)(0,2)
depends on D(2)(0,0) and D(0)(0,2)

:
d(1)(0,4)=min(d(0)(0,4),d(2)(0,0)+d(0)(0,4))d(1)(0,5)=min(d(0)(0,5),d(2)(0,0)+d(0)(0,5))d(1)(1,4)=min(d(0)(1,4),d(2)(1,0)+d(0)(0,4))d(1)(1,5)=min(d(0)(1,5),d(2)(1,0)+d(0)(0,5))d(2)(0,4)=min(d(1)(0,4),d(2)(0,1)+d(1)(1,4))d(2)(0,5)=min(d(1)(0,5),d(2)(0,1)+d(1)(1,5))d(2)(1,4)=min(d(1)(1,4),d(2)(1,1)+d(1)(1,4))d(2)(1,5)=min(d(1)(1,5),d(2)(1,1)+d(1)(1,5))

Phase 3: other blocks.

In the k
-th round, it computes all D(k⋅B)(h1,h2) where h1,h2≠k−1

.

The result of these blocks depends on the result from phase 2 and itself.

For instance, in the 1st round, the result of D(2)(1,2)
depends on D(2)(1,0) and D(2)(0,2)

:
d(1)(2,4)=min(d(0)(2,4),d(2)(2,0)+d(2)(0,4))d(1)(2,5)=min(d(0)(2,5),d(2)(2,0)+d(2)(0,5))d(1)(3,4)=min(d(0)(3,4),d(2)(3,0)+d(2)(0,4))d(1)(3,5)=min(d(0)(3,5),d(2)(3,0)+d(2)(0,5))d(2)(2,4)=min(d(1)(2,4),d(2)(2,1)+d(2)(1,4))d(2)(2,5)=min(d(1)(2,5),d(2)(2,1)+d(2)(1,5))d(2)(3,4)=min(d(1)(3,4),d(2)(3,1)+d(2)(1,4))d(2)(3,5)=min(d(1)(3,5),d(2)(3,1)+d(2)(1,5))

![image](https://user-images.githubusercontent.com/36917138/139832696-856fddc8-83bb-4f79-b5f4-12eff79fe164.png)
