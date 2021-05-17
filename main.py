"""
Demonstrates the algorithm for solving linear systems by Harrow, Hassidim,
Lloyd (HHL).

The HHL algorithm solves a system of linear equations, specifically equations
of the form Ax = b, where A is a Hermitian matrix, b is a known vector, and
x is the unknown vector. To solve on a quantum system, b must be rescaled to
have magnitude 1, and the equation becomes:

|x> = A**-1 |b> / || A**-1 |b> ||

The algorithm uses 3 sets of qubits: a single ancilla qubit, a register (to
store eigenvalues of A), and memory qubits (to store |b> and |x>). The
following are performed in order:
1) Quantum phase estimation to extract eigenvalues of A
2) Controlled rotations of ancilla qubit
3) Uncomputation with inverse quantum phase estimation

For details about the algorithm, please refer to papers in the
REFERENCE section below. The following description uses variables defined
in the HHL paper.

This example is an implementation of the HHL algorithm for arbitrary 2x2
Hermitian matrices. The output of the algorithm are the expectation values
of Pauli observables of |x>. Note that the accuracy of the result depends
on the following factors:
* Register size
* Choice of parameters C and t

The result is perfect if
* Each eigenvalue of the matrix is in the form

  2π/t * k/N,

  where 0≤k<N, and N=2^n, where n is the register size. In other words, k is a
  value that can be represented exactly by the register.
* C ≤ 2π/t * 1/N, the smallest eigenvalue that can be stored in the circuit.

The result is good if the register size is large enough such that for every
pair of eigenvalues, the ratio can be approximated by a pair of possible
register values. Let s be the scaling factor from possible register values to
eigenvalues. One way to set t is

t = 2π/sN

For arbitrary matrices, because properties of their eigenvalues are typically
unknown, parameters C and t are fine-tuned based on their condition number.


=== REFERENCE ===
Harrow, Aram W. et al. Quantum algorithm for solving linear systems of
equations (the HHL paper)
https://arxiv.org/abs/0811.3171

Coles, Eidenbenz et al. Quantum Algorithm Implementations for Beginners
https://arxiv.org/abs/1804.03719

=== CIRCUIT ===
Example of circuit with 2 register qubits.

(0, 0): ─────────────────────────Ry(θ₄)─Ry(θ₁)─Ry(θ₂)─Ry(θ₃)──────────────M──
                     ┌──────┐    │      │      │      │ ┌───┐
(1, 0): ─H─@─────────│      │──X─@──────@────X─@──────@─│   │─────────@─H────
           │         │QFT^-1│    │      │      │      │ │QFT│         │
(2, 0): ─H─┼─────@───│      │──X─@────X─@────X─@────X─@─│   │─@───────┼─H────
           │     │   └──────┘                           └───┘ │       │
(3, 0): ───e^iAt─e^2iAt───────────────────────────────────────e^-2iAt─e^-iAt─

Note: QFT in the above diagram omits swaps, which are included implicitly by
reversing qubit order for phase kickbacks.
"""

import math
import numpy as np
import sympy
import cirq


class PhaseEstimation(cirq.Gate):
    """
    A gate for Quantum Phase Estimation.

    unitary is the unitary gate whose phases will be estimated.
    The last qubit stores the eigenvector; all other qubits store the
    estimated phase, in big-endian.
    """

    def __init__(self, num_qubits, unitary, memory_size):
        self._num_qubits = num_qubits
        self.U = unitary
        self.memory_size = memory_size

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        qubits = list(qubits)
        yield cirq.H.on_each(*qubits[:-self.memory_size])
        yield PhaseKickback(self.num_qubits(), self.U, self.memory_size)(*qubits)
        yield cirq.qft(*qubits[:-self.memory_size], without_reverse=True) ** -1


class HamiltonianSimulation1(cirq.EigenGate, cirq.SingleQubitGate):
    """
    A gate that represents e^iAt.

    This EigenGate + np.linalg.eigh() implementation is used here
    purely for demonstrative purposes. If a large matrix is used,
    the circuit should implement actual Hamiltonian simulation,
    by using the linear operators framework in Cirq for example.
    """

    def __init__(self, A, t, exponent=1.0):
        cirq.SingleQubitGate.__init__(self)
        cirq.EigenGate.__init__(self, exponent=exponent)
        self.A = A
        self.t = t
        ws, vs = np.linalg.eigh(A)
        self.eigen_components = []
        for w, v in zip(ws, vs.T):
            theta = w * t / math.pi
            P = np.outer(v, np.conj(v))
            self.eigen_components.append((theta, P))

    def _with_exponent(self, exponent):
        return HamiltonianSimulation1(self.A, self.t, exponent)

    def _eigen_components(self):
        return self.eigen_components


class HamiltonianSimulation2(cirq.EigenGate, cirq.TwoQubitGate):
    """
    A gate that represents e^iAt.

    This EigenGate + np.linalg.eigh() implementation is used here
    purely for demonstrative purposes. If a large matrix is used,
    the circuit should implement actual Hamiltonian simulation,
    by using the linear operators framework in Cirq for example.
    """

    def __init__(self, A, t, exponent=1.0):
        cirq.TwoQubitGate.__init__(self)
        cirq.EigenGate.__init__(self, exponent=exponent)
        self.A = A
        self.t = t
        ws, vs = np.linalg.eigh(A)
        self.eigen_components = []
        for w, v in zip(ws, vs.T):
            theta = w * t / math.pi
            P = np.outer(v, np.conj(v))
            self.eigen_components.append((theta, P))

    def _with_exponent(self, exponent):
        return HamiltonianSimulation2(self.A, self.t, exponent)

    def _eigen_components(self):
        return self.eigen_components


def HamiltonianSimulation(A, t):
    if A.shape[0] == 2:
        sim = HamiltonianSimulation1(A, t)
    elif A.shape[0] == 4:
        sim = HamiltonianSimulation2(A, t)

    return sim


class PhaseKickback(cirq.Gate):
    """
    A gate for the phase kickback stage of Quantum Phase Estimation.

    It consists of a series of controlled e^iAt gates with the memory qubit as
    the target and each register qubit as the control, raised
    to the power of 2 based on the qubit index.
    unitary is the unitary gate whose phases will be estimated.
    """

    def __init__(self, num_qubits, unitary, memory_size):
        super(PhaseKickback, self)
        self._num_qubits = num_qubits
        self.U = unitary
        self.memory_size = memory_size

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        qubits = list(qubits)
        memory = qubits[-self.memory_size:]
        work_q = qubits[:-self.memory_size]

        for i, qubit in enumerate(work_q):
            yield cirq.ControlledGate(self.U ** (2 ** i))(qubit, *memory)


class EigenRotation(cirq.Gate):
    """
    EigenRotation performs the set of rotation on the ancilla qubit equivalent
    to division on the memory register by each eigenvalue of the matrix. The
    last qubit is the ancilla qubit; all remaining qubits are the register,
    assumed to be big-endian.

    It consists of a controlled ancilla qubit rotation for each possible value
    that can be represented by the register. Each rotation is a Ry gate where
    the angle is calculated from the eigenvalue corresponding to the register
    value, up to a normalization factor C.
    """

    def __init__(self, num_qubits, C, t):
        super(EigenRotation, self)
        self._num_qubits = num_qubits
        self.C = C
        self.t = t
        self.N = 2 ** (num_qubits - 1)

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        for k in range(self.N):
            kGate = self._ancilla_rotation(k)

            # xor's 1 bits correspond to X gate positions.
            xor = k ^ (k - 1)

            for q in qubits[-2::-1]:
                # Place X gates
                if xor % 2 == 1:
                    yield cirq.X(q)
                xor >>= 1

                # Build controlled ancilla rotation
                kGate = cirq.ControlledGate(kGate)

            yield kGate(*qubits)

    def _ancilla_rotation(self, k):
        if k == 0:
            k = self.N
        theta = 2 * math.asin(self.C * self.N * self.t / (2 * math.pi * k))
        return cirq.ry(theta)


def hhl_circuit(A, C, t, register_size, *input_prep_gates):
    """
    Constructs the HHL circuit.

    Args:
        A: The input Hermitian matrix.
        C: Algorithm parameter, see above.
        t: Algorithm parameter, see above.
        register_size: The size of the eigenvalue register.
        memory_basis: The basis to measure the memory in, one of 'x', 'y', 'z'.
        input_prep_gates: A list of gates to be applied to |0> to generate the
            desired input state |b>.

    Returns:
        The HHL circuit. The ancilla measurement has key 'a' and the memory
        measurement is in key 'm'.  There are two parameters in the circuit,
        `exponent` and `phase_exponent` corresponding to a possible rotation
        applied before the measurement on the memory with a
        `cirq.PhasedXPowGate`.
    """

    ancilla = cirq.LineQubit(0)
    # to store eigenvalues of the matrix
    register = [cirq.LineQubit(i + 1) for i in range(register_size)]

    # to store input and output vectors
    memory_size = int(np.log2(A.shape[0]))

    memory = [cirq.LineQubit(register_size + 1 + i) for i in range(memory_size)]


    c = cirq.Circuit()
    hs = HamiltonianSimulation(A, t)
    pe = PhaseEstimation(register_size + memory_size, hs, memory_size)
    c.append([gate(*memory) for gate in input_prep_gates])
    c.append(
        [
            pe(*(register + memory)),
            EigenRotation(register_size + 1, C, t)(*(register + [ancilla])),
            pe(*(register + memory)) ** -1,
            cirq.measure(ancilla, key='a'),
        ]
    )

    c.append(
        [
            cirq.measure(ancilla, *memory, key='m'),
        ]
    )

    return c


def simulate(circuit, A):
    global results
    
    simulator = cirq.Simulator()

    results = simulator.run(circuit, repetitions=200000)

    h = results.histogram(key='m')

    import math

    memory_size = int(np.log2(A.shape[0]))

    sol = dict(h.items())
    x = []
    for i in range(2**memory_size, 2**(memory_size+1)):
        try:
            x.append(sol[i])
        except:
            x.append(0)

    for i in range(len(x)):
        print(i, math.sqrt(x[i] / sum(x)))



def main():
    """
    Simulates HHL with matrix input, and outputs Pauli observables of the
    resulting qubit state |x>.
    Expected observables are calculated from the expected solution |x>.
    """

    A = np.array(
        [
            [5, -2, 0, 0],
            [-2, 1, 0, 0],
            [0, 0, 5, -2],
            [0, 0, -2, 1]
        ]
    )

    b = np.array([[1], [0], [0], [0]])
    sol = np.linalg.inv(A) @ b
    sol = sol / np.linalg.norm(sol)
    print('Classical solution')
    print(sol)

    L, v = np.linalg.eigh(A)

    ratio = max(L) / min(L)

    register_size = math.ceil(math.log2(ratio))
    print('Register size', register_size)

    t = 0.5723 #0.358166 * math.pi
    
    input_prep_gates = []#[cirq.ry(2 * -0.3948)]


    # Set C to be the smallest eigenvalue that can be represented by the
    # circuit.
    C = 2 * math.pi / (2 ** register_size * t)

    # Simulate circuit
    print("Results: ")
    simulate(hhl_circuit(A, C, t, register_size, *input_prep_gates), A)


if __name__ == '__main__':
    main()
