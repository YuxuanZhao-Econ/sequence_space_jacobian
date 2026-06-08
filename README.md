# Sequence-Space Jacobian in Julia

This repository contains Julia notebooks for learning the Sequence-Space Jacobian (SSJ) method from the ground up. The goal is not to provide the fastest implementation, but to make the algorithm transparent: model equations, DAG blocks, residual maps, Jacobians, impulse responses, news shocks, and nonlinear perfect-foresight transitions are all written out explicitly.

## Contents

- `RBC.ipynb`

  A representative-agent RBC model solved with a minimal SSJ implementation. The notebook covers steady state computation, DAG construction, sequence-space residuals, finite-difference Jacobians, general-equilibrium responses, and TFP shock experiments.

- `KS1998.ipynb`

  A Krusell-Smith style heterogeneous-agent model with idiosyncratic income risk. The notebook uses EGM for the household problem, distribution iteration for aggregation, and SSJ methods to compute transition dynamics and impulse responses.

- `SSJ_Function.jl`

  Shared helper routines used by the notebooks, including finite-difference Jacobians, Newton solving, DAG visualization, Rouwenhorst discretization, EGM, distribution iteration, and steady-state tools.

## Method

Both notebooks follow the same sequence-space logic:

1. Represent the model as a DAG of economic blocks.
2. Choose unknown sequences and target residuals.
3. Build the reduced residual system

   $$
   H(U,Z)=0.
   $$

4. Linearize around the steady state:

   $$
   H_U dU + H_Z dZ = 0.
   $$

5. Solve for the equilibrium response of unknowns:

   $$
   U_Z = -H_U^{-1}H_Z.
   $$

6. Recover the responses of other model variables by feeding the solved unknown paths back through the model blocks.


