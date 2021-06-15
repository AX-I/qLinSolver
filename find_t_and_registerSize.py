# Helper for HHL algorithm

import math
import sys

import numpy as np


def find_t_and_registerSize(A):

    lamda, vs = np.linalg.eigh(A)

    err_threshold_for_t = 0.05
    register_size_max   = 12

    for r_size in range(register_size_max):
        delta = choose_delta(r_size)

        t = choose_t(lamda, r_size, err_threshold_for_t, delta)
        if t < 5.0e30:
            return t, r_size

        if r_size > 10:
            print("try register_size=", r_size);

    if t > 5.0e30:
        print("Solutions for the equations are unreliable with the current parameters:");
        sys.exit(1)

    return t, register_size_max


def choose_delta(register_size):
    if   register_size <= 3:
         delta = 1
    elif register_size == 4:
         delta = 2
    elif register_size == 5:
         delta = 4
    elif register_size <  9:
         delta = 0.1 * 2**register_size
    elif register_size < 11:
         delta = 0.05 * 2**register_size
    elif register_size < 13:
         delta = 0.02 * 2**register_size
    elif register_size < 15:
         delta = 0.01 * 2**register_size
    elif register_size < 17:
         delta = 0.004 * 2**register_size
    elif register_size < 19:
         delta = 0.002 * 2**register_size
    elif register_size < 21:
         delta = 0.001 * 2**register_size
    else:
         delta = 0.0004 * 2**register_size

    return int(delta)


def choose_t(lamda, register_size, err_threshold, delta):

    # This function covers 2 cases of eigenvalues:
    #     case 1: all eigenvalues are positive
    #     case 2: some eigenvalues are positive

    # For case 3: if all eigenvalues are negative, then re-run the program with -A

    # delta: number of grids (out of N grids) as a gap:
    #        as gap between lamda_max and 2*pi  (if all eigenvalues are positive)
    #        as gap between positive and negative eigenvalues (otherwise)

    # local threshold - a hyper-parameter:
    # max_quantization_err = 0.3,  e.g., eigenvalue of 16.3 or 15.7 becomes binary value of 16.
    max_quantization_err = 0.3

    # determine local criteria of round off error
    # err_threshold:  quantization error relative to lamda range
    # err_threshold2: quantization error relative to 1 grid in binary grides of lamda

    N = 2**register_size
    err_threshold2 = err_threshold * N
    if  err_threshold2 > max_quantization_err:
        err_threshold2 = max_quantization_err

    # ------ determine t_min, t_max

    lambda_max   = max(lamda)
    lambda_min   = min(lamda)
    lambda_min_p = min(x for x in lamda if x > 0)
    if  lambda_min > 0:
        lambda_min = 0
    if  lambda_max < 0:
        print("Program exit: all eigenvalues are negative.")
        print("Supply -A to the program, re-run the program, and treat the solution as -x.")
        sys.exit(1)

    t_max = (2 * math.pi *(N-delta))/(N*(lambda_max - lambda_min))
    t_min = (2 * math.pi           )/(N* lambda_min_p)

    if t_max < t_min:
         # print("WARNING: Please increase register_size")
         return 8.88888e30

    #------ end of determining: t_min, t_max


    n_steps = 800
    delta_t = (t_max - t_min)/n_steps

    if abs(delta_t) < (1.0e-20):
        n_steps = 1
    elif delta_t < 0:
        n_steps = 0

    # err_min  is the        lowest error as t sweeps,  corresponding counter is i_best
    # err-min2 is the second lowest error as t sweeps,  corresponding counter is i_best2

    err_min  = 99.0
    err_min2 = 999999.0
    i_best = 0
    i_best2 = 0
    t = t_min
    for i in range(n_steps):

        err = 0
        for k in range(len(lamda)):
            tmp_lamda = lamda[k]
            if  tmp_lamda < 0:
                tmp_lamda += (2 * math.pi)/t

            tmp   = (t * tmp_lamda * N)/(2 * math.pi)
            tmp2  = tmp % 1
            if  tmp2 > 0.5:
                tmp2 = 1 - tmp2

            err += tmp2 * tmp2

            if tmp > (N - delta + 0.499):
                err += 99999999.99

        err = math.sqrt(err / k)

        #print("err=", err, "tmp=", tmp)

        if   err < err_min:
             err_min2 = err_min
             i_best2  = i_best
             err_min  = err
             i_best   = i

        elif err < err_min2:
             err_min2 = err
             i_best2  = i

        if i % 100 == 0:
             if err_min < err_threshold:
                 # If we are now within the first 200 steps of t, simply choose the best t
                 # If we are now beyond 200 steps, we can choose the 2nd best t if the 2nd best happened in much earlier steps.

                 if (i > 200) and ((i_best - i_best2) > 100) and (err_min2 < 1.2 * err_threshold):
                     t_best2 = t_min + i_best2 * delta_t
                     #print("set t = %10.4e" % t_best2, "(i =", i_best2, "/", n_steps,")...")
                     return t_best2
                 else:
                     t_best = t_min + i_best * delta_t
                     #print("set t = %10.4e" % t_best,  "(i =", i_best,   "/", n_steps,")....")
                     return t_best
        t += delta_t

    return 9.9999e30
