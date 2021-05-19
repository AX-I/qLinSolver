"""
Demonstrates the algorithm for solving linear systems by Harrow, Hassidim,
Lloyd (HHL).
"""

import math
import numpy as np
import sympy
import cirq
import cirq.ops.raw_types as raw_types
import abc

class AnyQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly 'any' qubits."""

    def __init__(self, num_qubits):
        super().__init__()
        self.num_qubits = num_qubits

    def _num_qubits_(self) -> int:
        return self.num_qubits



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




class HamiltonianSimulation(cirq.EigenGate, AnyQubitGate):
    """
    A gate that represents e^iAt.

    This EigenGate + np.linalg.eigh() implementation is used here
    purely for demonstrative purposes. If a large matrix is used,
    the circuit should implement actual Hamiltonian simulation,
    by using the linear operators framework in Cirq for example.
    """

    def __init__(self, A, t, exponent=1.0):
        num_qubits = int(np.log2(A.shape[0]))

        AnyQubitGate.__init__(self, num_qubits)
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
        return HamiltonianSimulation(self.A, self.t, exponent)

    def _eigen_components(self):
        return self.eigen_components


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

    def __init__(self, num_qubits, C, t, k_border):
        super(EigenRotation, self)
        self._num_qubits = num_qubits
        self.C = C
        self.t = t
        self.N = 2 ** (num_qubits - 1)
        self.k_border = k_border

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

        if (k > self.k_border) and (k != self.N):
            theta = 2 * math.asin(self.C * self.N * self.t / (2 * math.pi * (k - self.N)))
        else:
            theta = 2 * math.asin(self.C * self.N * self.t / (2 * math.pi * k))
        return cirq.ry(theta)


class InputPrepGates(cirq.Gate):

    def __init__(self, b):
        super(InputPrepGates, self)

        self._num_qubits = int(np.log2(len(b)))

        self.b = b

    def num_qubits(self):
        return self._num_qubits

    def _recurse(self, b, qubits, depth=0):
        k = len(b) // 2

        if depth == 0:
            yield rotTheta(b[:k], b[k:], None, qubits[0], control=False)

        if k == 1:
            return None

        yield cirq.X(qubits[0])

        yield rotTheta(b[:k//2], b[k//2:k], qubits[0], qubits[1])

        yield cirq.X(qubits[0])

        if k > 2:
            # Recurse
            yield self._recurse(b[:k], qubits[1:], depth + 1)


        yield rotTheta(b[k:k+k//2], b[k+k//2:], qubits[0], qubits[1])

        if k > 2:
            # Recurse
            yield self._recurse(b[k:], qubits[1:], depth + 1)

    def _decompose_(self, qubits):
        b = self.b
        qubits = list(qubits)

        return self._recurse(b, qubits)


def rotTheta(b_den, b_num, qubitA, qubitB, control=True):

    if type(b_den) == np.ndarray and len(b_den) > 1:
        den = sum(x**2 for x in b_den)
        num = sum(x**2 for x in b_num)
    else:
        den = b_den
        num = b_num

    if den == 0:
        if num == 0:
            return cirq.ry(0)(qubitB)
        sign_num = 1 if num > 0 else -1

        if control:
            return cirq.ControlledGate(cirq.ry(sign_num * 3.1415))(qubitA, qubitB)
        else:
            return cirq.ry(sign_num * 3.1415)(qubitB)

    else:
        if type(b_den) == np.ndarray and len(b_den) > 1:
            theta = np.arctan(np.sqrt(num / den))
        else:
            theta = np.arctan(num / den)

        if control:
            return cirq.ControlledGate(cirq.ry(2 * theta))(qubitA, qubitB)
        else:
            return cirq.ry(2 * theta)(qubitB)


class InnerProduct(cirq.Gate):
    def __init__(self, mem_size):
        super(InnerProduct, self)
        self.mem_size = mem_size
        self._num_qubits = 1 + 2 * mem_size

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        anc = qubits[0]
        mem1 = qubits[1 : 1 + self.mem_size]
        mem2 = qubits[1 + self.mem_size:]

        yield cirq.H(anc)

        for i in range(self.mem_size):
            yield cirq.ControlledGate(cirq.SWAP)(anc, mem1[i], mem2[i])

        yield cirq.H(anc)


def hhl_circuit(A, C, t, register_size, b, k_border, select):
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

    if select is not None:
        mem_select = [cirq.LineQubit(memory_size + register_size + 1 + i) for i in range(memory_size)]
        ancilla_select = cirq.LineQubit(2*memory_size + register_size + 1)

    c = cirq.Circuit()
    hs = HamiltonianSimulation(A, t)
    pe = PhaseEstimation(register_size + memory_size, hs, memory_size)

    c.append([
        InputPrepGates(b)(*memory)
    ])

    if select is not None:
        c.append([
            InputPrepGates(select)(*mem_select)
        ])

    c.append(
        [
            pe(*(register + memory)),
            EigenRotation(register_size + 1, C, t, k_border)(*(register + [ancilla])),
            pe(*(register + memory)) ** -1,
            cirq.measure(ancilla, key='a'),
        ]
    )

    if select is not None:
        c.append([
            InnerProduct(len(memory))(ancilla_select, *memory, *mem_select),
            cirq.measure(ancilla, ancilla_select, key='s')
        ])

    else:
        c.append([
            cirq.measure(ancilla, *memory, key='m'),
        ])

    return c


def simulate(circuit, A):
    global results
    import math

    simulator = cirq.Simulator()

    results = simulator.run(circuit, repetitions=5 * 10**5)

    try:
        h = results.histogram(key='m')
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
    except KeyError:
        print('Not measuring solution')

    try:
        s = results.histogram(key='s')
        prob = s[2] / (s[2] + s[3])
        inn = math.sqrt(2 * prob - 1)
        print('Inner product', inn)
    except KeyError:
        print('Not measuring select')


def main():
    """
    Simulates HHL with matrix input, and outputs Pauli observables of the
    resulting qubit state |x>.
    Expected observables are calculated from the expected solution |x>.
    """
    select = None

    A = np.array(
        [
            [5, 0],
            [0, -1]
        ]
    )
##    A = np.array(
##        [
##            [5, -2, 0, 0],
##            [-2, 1, 0, 0],
##            [0, 0, 5, -2],
##            [0, 0, -2, 1]
##        ]
##    )
##    A = np.array(
##        [
##            [5, -2, 0, 0, 0, 0, 0, 0],
##            [-2, 1, 0, 0, 0, 0, 0, 0],
##            [0, 0, 5, -2, 0, 0, 0, 0],
##            [0, 0, -2, 1, 0, 0, 0, 0],
##            [0, 0, 0, 0, 5, -2, 0, 0],
##            [0, 0, 0, 0, -2, 1, 0, 0],
##            [0, 0, 0, 0, 0, 0, 5, -2],
##            [0, 0, 0, 0, 0, 0, -2, 1]
##        ]
##    )

    b = np.array([[12], [-5]])
    #b = np.array([[12], [-5], [1], [0], [0], [0], [0], [0]])
    #b = np.array([[1], [0], [0], [0], [0], [0], [0], [0]])

    select = np.array([[1], [0]])

    # ==== ==== ==== ==== End of User Input ==== ==== ==== ====


    L, v = np.linalg.eigh(A)

    from find_t_and_registerSize import find_t_and_registerSize
    t, register_size = find_t_and_registerSize(A)

    print('t', t)
    print('Register size', register_size)

    if select is not None:
        select = select / np.linalg.norm(select)

    # k_border is the overflow limit between positive and negative eigenvalues
    k_border = 1 + math.floor((2**register_size * max(L) * t / (2*3.1415)))
    print('border', k_border)

    # Set C to be the smallest eigenvalue that can be represented by the
    # circuit.
    C = 2 * math.pi / (2 ** register_size * t)

    sol = np.linalg.inv(A) @ b
    sol = sol / np.linalg.norm(sol)
    print('Classical solution')
    print(sol)

    # Simulate circuit
    print("Results: ")
    simulate(hhl_circuit(A, C, t, register_size, b, k_border, select), A)


if __name__ == '__main__':
    main()
