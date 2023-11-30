# EE514-final-project
Daniel and Aakash final project

## **Option 1: optimized encoded CNOT for N=4 through reinforcement learning**

This is the fixed ansatz with l=30 (30 fixed layers):

![Fixed QNN Architecture](./pics/fixed_QNN_architecture.png)


The unitary in the picture above with p=+1 or p=-1 represents the swap operator with some global phase. The unitary with p=0 represents the identity operator on two physical qubits.

I shall use the fixed QNN architecture in the picture above (with 30 fixed layers) and use reinforcement learning to incentivize using more parameters with p=0 (identity operators) which means less number of non-identity parametrized  operators. This will allow us to find a compact version of the encoded CNOT operation on two logical qubits.

I am looking to use a reinforcement learning methods which are commonly used for continuous action spaces. Some examples are: Deep Q Networks (DQN) or policy gradient methods (e.g., Proximal Policy Optimization, Trust Region Policy Optimization), but will look into other algorithms.



## **Option 2: doing QEC for t=1 errors on VQE problem**

For VQE not much research has been put forth on mitigating error but as per the research showing the effects of error on the VQE a error rate of 1% can cause result in failure to aten the lowest energy (eigenvalue). Error rate less that 0.18% can provide satisfactory results givving close enough convergence rate.

Proposed idea is to use the Error correcting code of [[5,1,3]] or [[4,2,2]] to correct the Pauli Errors in the Ansatz Circuit.

I propose to create a subroutine before each rotation operation in the ansatz in sequence of encoding -> noise -> detection -> correction -> decoding

Its easier to apply for less qubit circuit with less depth.

Custom codes need to be made catering to specific ansatz with more qubits and larger circuit depth to not hamper the computational time.

