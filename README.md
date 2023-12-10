# EE514-final-project
Daniel and Aakash final project


## **How to run the code:**
#### 1. Make sure you have installed miniconda(https://docs.conda.io/projects/miniconda/en/latest/) through terminal and is recognizable by your environment variables
#### 2. Open your terminal, then `cd <into-the-root-of-this-folder>`, then run `bash runCode.sh` (you only need to do this once if you've done it right). That last script will create an virtual environment called `qrl` which will store all your dependencies
#####    - in case anything goes wrong during running the file. Use `conda activate qrl`, then `pip install <some-dependency>` as the errors come up. You will see what dependencies are essential when being used or not by the python notebooks.
#### 3. Open your VisualStudio to the root directory of this folder, and open `fixed_QNN.ipynb`. Run the entire notebook using the virtual environment `qrl`




## **Project Description: Optimized encoded CNOT for N=3 through reinforcement learning**

This is the fixed ansatz with l=30 (30 fixed layers):

![Fixed QNN Architecture](./pics/fixed_QNN_architecture.png)


The unitary in the picture above with p=+1 or p=-1 represents the swap operator with some global phase. The unitary with p=0 represents the identity operator on two physical qubits.

I shall use the fixed QNN architecture in the picture above (with 30 fixed layers) and use reinforcement learning to incentivize using more parameters with p=0 (identity operators) which means less number of non-identity parametrized  operators. This will allow us to find a compact version of the encoded CNOT operation on two logical qubits.

I am looking to use a reinforcement learning methods which are commonly used for continuous action spaces. Some examples are: Deep Q Networks (DQN) or policy gradient methods (e.g., Proximal Policy Optimization, Trust Region Policy Optimization), but will look into other algorithms.

