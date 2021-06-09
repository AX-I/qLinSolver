"""
Demonstrates the algorithm for solving linear systems by Harrow, Hassidim,
Lloyd (HHL).
"""

import math
import numpy as np
import cirq

from HHL_Methods import precondition, readInput

from HHL_Methods import PhaseEstimation, HamiltonianSimulation, PhaseKickback
from HHL_Methods import EigenRotation, InputPrepGates
from HHL_Methods import InnerProduct


def hhl_circuit(A, C, t, register_size, b, k_border,
                select, getMagnitude):
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

    # ancilla used for Swap gate
    anc_swap = cirq.LineQubit(100)

    # y qubits for use in MGate and Swap
    anc_y = [cirq.LineQubit(200+i) for i in range(len(memory))]

    c = cirq.Circuit()
    hs = HamiltonianSimulation(A, t)
    pe = PhaseEstimation(register_size, memory_size, hs)

    c.append([
        InputPrepGates(b)(*memory)
    ])

    if select is not None:
        c.append([
            InputPrepGates(select)(*mem_select)
        ])

    c.append(
        [
            pe(*(register + memory + anc_y), anc_swap),
            EigenRotation(register_size + 1, C, t, k_border)(*(register + [ancilla])),
            pe(*(register + memory + anc_y), anc_swap) ** -1,
        ]
    )

    if select is not None:
        c.append([
            InnerProduct(len(memory))(ancilla_select, *memory, *mem_select),
            cirq.measure(ancilla, ancilla_select, key='s')
        ])

    elif not getMagnitude:
        c.append([
            cirq.measure(ancilla, *memory, key='m'),
        ])


    if getMagnitude:
        max_b = [abs(e) for e in b].index(max([abs(e) for e in b]))
        row = A[max_b]
        row_mag = np.linalg.norm(row)
        row_q = [cirq.LineQubit(2*memory_size + register_size + 2 + i) for i in range(memory_size)]
        row_anc = cirq.LineQubit(3*memory_size + register_size + 2)

        c.append([
            InputPrepGates(row)(*row_q),
            InnerProduct(len(memory))(row_anc, *row_q, *memory),
            cirq.measure(ancilla, row_anc, key='x')
        ])

    return c


def simulate(circuit, A, b, factors, reps):
    global results
    import math

    simulator = cirq.Simulator()

    results = simulator.run(circuit, repetitions=reps)

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

        x = np.array(x)
        x = np.sqrt(x) / np.sum(x) / np.array(factors)
        x = x / np.linalg.norm(x)
        for i in range(len(x)):
            print(i, x[i])

        return x

    except KeyError:
        print('Not measuring solution')

    try:
        s = results.histogram(key='s')
        prob = s[2] / (s[2] + s[3])
        inn = math.sqrt(2 * prob - 1)
        print('Inner product', inn)

    except KeyError:
        print('Not measuring select')

    try:
        x = results.histogram(key='x')
        prob = x[2] / (x[2] + x[3])
        inn = math.sqrt(max(0, 2 * prob - 1))
        max_b = [abs(e) for e in b].index(max([abs(e) for e in b]))
        row = A[max_b]
        row_mag = np.linalg.norm(row)
        sign = -1 if b[max_b] < 0 else 1
        print('Magnitude', sign * b[max_b] / (inn * row_mag))

    except KeyError:
        print('Not measuring magnitude')



def main():
    """
    Simulates HHL with matrix input, and outputs Pauli observables of the
    resulting qubit state |x>.
    Expected observables are calculated from the expected solution |x>.
    """
    select = None
    getMagnitude = False

    #A, b = readInput(open('A_in.txt'), open('B_in.txt'))

##    A = np.array(
##        [
##            [5, 0],
##            [0, -1]
##        ]
##    )
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

    A = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ]
    )
    b = np.array([[0, 0, 1, 0]]).T

    #b = np.array([[12], [-5], [1], [0]])
    #b = np.array([[12], [-5], [1], [0], [0], [0], [0], [0]])
    #b = np.array([[1], [0], [0], [0], [0], [0], [0], [0]])

    #select = np.array([[1], [0]])
    #getMagnitude = True

    # ==== ==== ==== ==== End of User Input ==== ==== ==== ====

    #A = A.astype('float')
    #b = b.astype('float')
    factors = [1] * len(A)
    #factors = precondition(A, b)


    L, v = np.linalg.eigh(A)
    print('Condition number', max(L) / min(L))

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

    for i in range(len(sol)):
        sol[i] /= factors[i]
        sol[i] = math.sqrt((sol[i] * sol[i].conjugate()).real)

    if getMagnitude:
        print('Classical magnitude')
        print(np.linalg.norm(sol))
    sol = sol / np.linalg.norm(sol)

    print('Classical solution')
    print(sol)
    if select is not None:
        print('Inner product')
        print(round((select.T @ sol)[0][0], 8))

    # Simulate circuit
    r = 5 * 10**5
    print('Repetitions:', r)
    print("Results: ")
    actual = simulate(hhl_circuit(A, C, t, register_size, b,
                         k_border, select, getMagnitude),
                      A, b, factors, reps=r)

    sol = sol.flatten()
    actual = actual.flatten()
    print('Relative error:', math.sqrt((sum((sol - actual) ** 2)).real))


if __name__ == '__main__':
    main()
