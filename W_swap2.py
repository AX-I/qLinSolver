# M gate with integer values, Z⊗Z⊗F matrix exp
# WSwapExp with imaginary GTG-1

import math
import numpy as np
import cirq
from W_swap import WGate

class ZZFExp(cirq.Gate):
    def __init__(self, r, t, exp=1.0):
        """Performs e^(-i(Z⊗Z⊗F)t)
        Qubits used:
        anc_swap (1), anc_zzf (1), w_sign (1), w (3*r)"""
        super(ZZFExp, self)
        self._num_qubits = 3*r + 3
        self.r = r
        self.t = t
        self.exp = exp

    def num_qubits(self):
        return self._num_qubits

    def __pow__(self, power):
        return ZZFExp(self.r, self.t, exp=power)

    def _decompose_(self, qubits):
        anc_swap = qubits[0]
        anc_zzf = qubits[1] # parity
        w_sign = qubits[2]
        qw = qubits[3:3+3*self.r]

        yield cirq.ControlledGate(cirq.X)(anc_swap, anc_zzf)
        yield cirq.ControlledGate(cirq.X)(w_sign, anc_zzf)

        for i in range(3*self.r):
            crot = cirq.ControlledGate(cirq.ZPowGate(exponent= self.exp * (2**i) * self.t/math.pi))
            yield crot(anc_zzf, qw[3*self.r - i - 1])

        yield cirq.X(anc_zzf)

        for i in range(3*self.r):
            crot = cirq.ControlledGate(cirq.ZPowGate(exponent= self.exp * (-(2**i)) * self.t/math.pi))
            yield crot(anc_zzf, qw[3*self.r - i - 1])

        yield cirq.X(anc_zzf)
        yield cirq.ControlledGate(cirq.X)(w_sign, anc_zzf)
        yield cirq.ControlledGate(cirq.X)(anc_swap, anc_zzf)


class MGateValue(cirq.Gate):
    def __init__(self, n, params, r):
        """params -> list of (row, value)
        Qubits used:
        memory (n), anc_y (n), w (3*r), w_sign (1)"""
        super(MGateValue, self)
        self._num_qubits = 2*n + 3*r + 1
        self.n = n
        self.r = r
        self.params = params

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        n = self.n
        qx = qubits[:n]
        qy = qubits[n:2*n]

        qw = qubits[2*n : 2*n + 3*self.r]
        w_sign = qubits[2*n + 3*self.r]

        for k in range(2**n):
            bin_x = ('{:'+str(n)+'b}').format(k)
            for i in range(len(bin_x)):
                if bin_x[i] != '1':
                    yield cirq.X(qx[i])


            bin_y = ('{:'+str(n)+'b}').format(self.params[k][0])
            for i in range(len(bin_y)):
                if bin_y[i] == '1':
                    cx = cirq.ControlledGate(cirq.X, num_controls=n)
                    yield cx(*qx, qy[i])


            bin_w = ('{:'+str(3*self.r)+'b}').format(abs(int(self.params[k][1].real)))
            for i in range(len(bin_w)):
                if bin_w[i] == '1':
                    cx = cirq.ControlledGate(cirq.X, num_controls=n)
                    yield cx(*qx, qw[i])

            if self.params[k][1] < 0:
                yield cirq.X(w_sign)


            for i in range(len(bin_x)):
                if bin_x[i] != '1':
                    yield cirq.X(qx[i])


class GGate(cirq.Gate):
    def __init__(self, exp=1):
        super(GGate, self)
        self._num_qubits = 1
        self.exp = exp
    def num_qubits(self):
        return 1
    def __pow__(self, p):
        return GGate(exp=p)
    def _unitary_(self):
        u = -1/math.sqrt(2) * np.array([[complex(0, 1), 1],
                                        [1, complex(0, 1)]])
        if self.exp == 1:
            return u
        elif self.exp == -1:
            return np.linalg.inv(u)


class WSwapExponentValue(cirq.Gate):
    def __init__(self, t, num_swaps, r, power=1.0, imaginary=False):
        """Qubits used:
        memory (num), anc_y (num), anc_swap (1), anc_zzf (1), w_sign (1), w (3*r)"""
        super(WSwapExponentValue, self)
        self.t = t
        self.r = r
        self.imaginary = imaginary
        self.power = power
        self.num_swaps = num_swaps
        self._num_qubits = num_swaps * 2 + 3 + 3*r

    def __pow__(self, exp):
        return WSwapExponentValue(self.t, self.num_swaps, self.r,
                                  power=exp, imaginary=self.imaginary)

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        anc_swap = qubits[2*self.num_swaps]
        qx = qubits[:self.num_swaps]
        qy = qubits[self.num_swaps:2*self.num_swaps]

        for i in range(self.num_swaps):
            yield WGate()(qx[i], qy[i])
            yield cirq.X(qy[i])

        for i in range(self.num_swaps):
            cx = cirq.ControlledGate(cirq.X, num_controls=2)
            yield cx(qx[i], qy[i], anc_swap)


        anc_zzf = qubits[2*self.num_swaps + 1]
        w_sign = qubits[2*self.num_swaps + 2]
        qw = qubits[2*self.num_swaps+3 : 2*self.num_swaps+3 + 3*self.r]

        if self.imaginary:
            yield GGate()(w_sign)

        yield ZZFExp(self.r, self.power * self.t)(anc_swap, anc_zzf, w_sign, *qw)

        if self.imaginary:
            yield GGate()(w_sign) ** -1


        for i in range(self.num_swaps-1, -1, -1):
            cx = cirq.ControlledGate(cirq.X, num_controls=2)
            yield cx(qx[i], qy[i], anc_swap)

        for i in range(self.num_swaps):
            yield cirq.X(qy[i])
            yield WGate()(qx[i], qy[i])

