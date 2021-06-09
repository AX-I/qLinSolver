"""
Methods for use in HHL
"""

import math
import numpy as np
import cirq
import cirq.ops.raw_types as raw_types
import abc

from W_swap import WGate, WSwap, MGate, WSwapExponent

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

    def __init__(self, reg_size, mem_size, unitary):
        self._num_qubits = reg_size + mem_size * 2 + 1
        self.U = unitary
        self.mem_size = mem_size
        self.reg_size = reg_size

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        qubits = list(qubits)
        yield cirq.H.on_each(*qubits[:self.reg_size])
        yield PhaseKickback(self.num_qubits(), self.U,
                            self.reg_size, self.mem_size)(*qubits)
        yield cirq.qft(*qubits[:self.reg_size], without_reverse=True) ** -1



class HamiltonianSimulation(cirq.Gate):
    """
    A gate that represents e^iAt. Based on Ahokas 2004.

    Qubits used:
    memory (mem_size), anc_y (mem_size), anc_swap (1)
    """

    def __init__(self, M, t, exponent=1.0):
        super(HamiltonianSimulation, self)

        nb = int(np.log2(M.shape[0]))

        self.M = M

        self._num_qubits = nb * 2 + 1

        self.num_qx = nb

        self.t = t
        self.exp = exponent

        self.params = []
        for i in range(M.shape[0]):
            self.params.append(
                np.where(M[:,i] != 0)[0][0]
            )

    def num_qubits(self):
        return self._num_qubits

    def __pow__(self, exp):
        return HamiltonianSimulation(self.M, self.t, exponent=exp)

    def _decompose_(self, qubits):
        num_swaps = self.num_qx

        qx = qubits[:self.num_qx]
        qy = qubits[self.num_qx:2*self.num_qx]
        anc = qubits[2*self.num_qx]

        yield MGate(self.num_qx, self.params)(*qx, *qy)
        yield (WSwapExponent(self.t, num_swaps)**self.exp) (*qx, *qy, anc)
        yield MGate(self.num_qx, self.params)(*qx, *qy)


class PhaseKickback(cirq.Gate):
    """
    A gate for the phase kickback stage of Quantum Phase Estimation.

    It consists of a series of controlled e^iAt gates with the memory qubit as
    the target and each register qubit as the control, raised
    to the power of 2 based on the qubit index.
    unitary is the unitary gate whose phases will be estimated.
    """

    def __init__(self, num_qubits, unitary, reg_size, mem_size, exp=1.0):
        super(PhaseKickback, self)
        self._num_qubits = num_qubits
        self.U = unitary
        self.mem_size = mem_size
        self.reg_size = reg_size
        self.exp = exp

    def num_qubits(self):
        return self._num_qubits

    def __pow__(self, exp):
        return PhaseKickback(self._num_qubits, self.U,
                             self.reg_size, self.mem_size, exp)

    def _decompose_(self, qubits):
        qubits = list(qubits)
        mem_qubits = qubits[self.reg_size:]
        work_q = qubits[:self.reg_size]

        for i, qubit in enumerate(work_q):
            yield cirq.ControlledGate(self.U ** (self.exp * 2 ** i))(qubit, *mem_qubits)


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

    def _recurseMag(self, b, qubits, depth=0):
        d = depth
        k = len(b) // 2

        if depth == 0:
            yield rotTheta(b[:k], b[k:], [], qubits[0], control=False)

        if k == 1:
            return None

        yield cirq.X(qubits[d])

        yield rotTheta(b[:k//2], b[k//2:k], qubits[:d+1], qubits[d+1])

        if k > 2:
            # Recurse
            yield self._recurseMag(b[:k], qubits, depth + 1)

        yield cirq.X(qubits[d])


        yield rotTheta(b[k:k+k//2], b[k+k//2:], qubits[:d+1], qubits[d+1])

        if k > 2:
            # Recurse
            yield self._recurseMag(b[k:], qubits, depth + 1)


    def _recursePhase(self, b, qubits, depth=0):
        d = depth
        k = len(b) // 2

        if depth == 0:
            yield rotPhase(b[:k], b[k:], [], qubits[0], control=False)

        if k == 1:
            return None

        yield cirq.X(qubits[d])

        yield rotPhase(b[:k//2], b[k//2:k], qubits[:d+1], qubits[d+1])

        if k > 2:
            # Recurse
            yield self._recursePhase(b[:k], qubits, depth + 1)

        yield cirq.X(qubits[d])


        yield rotPhase(b[k:k+k//2], b[k+k//2:], qubits[:d+1], qubits[d+1])

        if k > 2:
            # Recurse
            yield self._recursePhase(b[k:], qubits, depth + 1)

    def _decompose_(self, qubits):
        b = self.b
        qubits = list(qubits)

        yield self._recurseMag(b, qubits)

        yield self._recursePhase(b, qubits)


def rotTheta(b_den, b_num, qubitsA, qubitB, control=True):
    NC = len(qubitsA)

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
            crot = cirq.ControlledGate(cirq.ry(sign_num * 3.1415), num_controls=NC)
            return crot(*qubitsA, qubitB)
        else:
            return cirq.ry(sign_num * 3.1415)(qubitB)

    else:
        if type(b_den) == np.ndarray and len(b_den) > 1:
            theta = np.arctan(np.sqrt(num / den))
        else:
            theta = np.arctan(num / den)

        if control:
            crot = cirq.ControlledGate(cirq.ry(2 * theta), num_controls=NC)
            return crot(*qubitsA, qubitB)
        else:
            return cirq.ry(2 * theta)(qubitB)


def phase(z):
    return math.atan2(z.imag, z.real)

def rotPhase(b_den, b_num, qubitsA, qubitB, control=True):
    NC = len(qubitsA)

    if type(b_den) == np.ndarray and len(b_den) > 1:
        den = sum(phase(z) for z in b_den)
        num = sum(phase(z) for z in b_num)
    else:
        den = phase(b_den)
        num = phase(b_num)

    if den == 0:
        if num == 0:
            return cirq.rz(0)(qubitB)
        sign_num = 1 if num > 0 else -1

        if control:
            crot = cirq.ControlledGate(cirq.rz(sign_num * 3.1416), num_controls=NC)
            return crot(*qubitsA, qubitB)
        else:
            return cirq.rz(sign_num * 3.1416)(qubitB)
    else:
        theta = (num - den) / 2
        if control:
            crot = cirq.ControlledGate(cirq.rz(2 * theta), num_controls=NC)
            return crot(*qubitsA, qubitB)
        else:
            return cirq.rz(2 * theta)(qubitB)


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


# ==== End of quantum methods ==== #


def precondition(A, b):
    factors = []

    for i in range(A.shape[0]):
        factor = math.sqrt(A[i,i].real)
        A[i] = A[i] / factor
        A[:,i] = A[:,i] / factor
        b[i] = b[i] / factor

        factors.append(factor)

    return factors


def readInput(afile, bfile):
    N = 2 ** math.ceil(math.log2(max([int(line.split(' ')[0]) for line in afile])))
    afile.seek(0)

    A_out = np.zeros((N, N), 'complex')

    for line in afile:
        s = line.split(' ')
        h = int(s[0])
        w = int(s[1])
        val = complex(float(s[2]), float(s[3]))
        A_out[h, w] = val
        A_out[w, h] = val.conjugate()

    b_out = np.zeros((N, 1), 'complex')

    for line in bfile:
        s = line.split(' ')
        h = int(s[0])
        val = complex(float(s[1]), float(s[2]))
        b_out[h] = val

    return (A_out, b_out)


if __name__ == '__main__':
    pass
