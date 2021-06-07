# W gate and alternate swap
# Based on Ahokas 2004

import math
import numpy as np
import cirq

class MGate(cirq.Gate):
    """Takes a permutation matrix as parameter.
    Input x and ancilla, mutates ancilla into y coordinate."""

    def __init__(self, n, params):
        """params -> list of column numbers"""
        super(MGate, self)
        self._num_qubits = n*2
        self.n = n
        self.params = params

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        n = self.n
        qx = qubits[:n]
        qy = qubits[n:]

        for k in range(2**n):
            bin_x = ('{:'+str(n)+'b}').format(k)
            for i in range(len(bin_x)):
                if bin_x[i] != '1':
                    yield cirq.X(qx[i])

            bin_y = ('{:'+str(n)+'b}').format(self.params[k])
            for i in range(len(bin_y)):
                if bin_y[i] == '1':
                    cx = cirq.ControlledGate(cirq.X, num_controls=n)
                    yield cx(*qx, qy[i])

            for i in range(len(bin_x)):
                if bin_x[i] != '1':
                    yield cirq.X(qx[i])


class WGate(cirq.Gate):
    def __init__(self):
        super(WGate, self)
        self._num_qubits = 2

    def num_qubits(self):
        return self._num_qubits

    def _unitary_(self):
        s2 = math.sqrt(0.5)
        return np.array([[1, 0, 0, 0],
                         [0, s2, s2, 0],
                         [0, s2, -s2, 0],
                         [0, 0, 0, 1]])


class WSwap(cirq.Gate):
    """Swaps pairs of (x,y) qubits"""

    def __init__(self, num_swaps):
        super(WSwap, self)
        self.num_swaps = num_swaps
        self._num_qubits = num_swaps * 2 + 1

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        anc = qubits[-1]

        for i in range(self.num_swaps):
            yield WGate()(qubits[2*i], qubits[2*i + 1])

            yield cirq.X(qubits[2*i + 1])

        for i in range(self.num_swaps):
            cx = cirq.ControlledGate(cirq.X, num_controls=2)
            yield cx(qubits[2*i], qubits[2*i + 1], anc)

        yield cirq.Z(anc)

        for i in range(self.num_swaps-1, -1, -1):
            cx = cirq.ControlledGate(cirq.X, num_controls=2)
            yield cx(qubits[2*i], qubits[2*i + 1], anc)

        for i in range(self.num_swaps):
            yield cirq.X(qubits[2*i + 1])
            yield WGate()(qubits[2*i], qubits[2*i + 1])


class WSwapExponent(cirq.Gate):
    """e^(i WSWAP t)"""

    def __init__(self, t, num_swaps, power=1.0):
        super(WSwapExponent, self)
        self.t = t
        self.power = power
        self.num_swaps = num_swaps
        self._num_qubits = num_swaps * 2 + 1

    def __pow__(self, exp):
        return WSwapExponent(self.t, self.num_swaps, power=exp)

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        anc = qubits[-1]

        for i in range(self.num_swaps):
            yield WGate()(qubits[2*i], qubits[2*i + 1])
            yield cirq.X(qubits[2*i + 1])

        for i in range(self.num_swaps):
            cx = cirq.ControlledGate(cirq.X, num_controls=2)
            yield cx(qubits[2*i], qubits[2*i + 1], anc)

        yield cirq.rz(self.power * 2 * self.t)(anc)

        for i in range(self.num_swaps-1, -1, -1):
            cx = cirq.ControlledGate(cirq.X, num_controls=2)
            yield cx(qubits[2*i], qubits[2*i + 1], anc)

        for i in range(self.num_swaps):
            yield cirq.X(qubits[2*i + 1])
            yield WGate()(qubits[2*i], qubits[2*i + 1])
