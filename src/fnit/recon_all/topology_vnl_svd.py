"""Source-order float32 3x3 SVD inverse for first topology curvature fit.

Port of ITK 5.3 VNL vnl_svd<float> (LINPACK ssvdc, job=21) and FreeSurfer
8.2 MatrixSVDInverse. Inputs and outputs use NumPy float32 arrays; no native
FreeSurfer, ITK, or BLAS library is loaded by this module.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit
def f(x):
    return np.float32(x)


@njit
def add(x, y):
    return f(f(x) + f(y))


@njit
def sub(x, y):
    return f(f(x) - f(y))


@njit
def mul(x, y):
    return f(f(x) * f(y))


@njit
def div(x, y):
    return f(f(x) / f(y))


@njit
def norm(values):
    if len(values) == 1:
        return f(abs(values[0]))
    scale = f(0)
    ssq = f(1)
    for x in values:
        if x != 0:
            ax = f(abs(x))
            if scale < ax:
                r = div(scale, ax)
                ssq = add(mul(ssq, mul(r, r)), 1)
                scale = ax
            else:
                r = div(ax, scale)
                ssq = add(ssq, mul(r, r))
    return f(np.float64(scale) * math.sqrt(np.float64(ssq)))


@njit
def dot(x, y):
    t = f(0)
    for a, b in zip(x, y):
        t = add(t, mul(a, b))
    return t


@njit
def rotg(sa, sb):
    roe = sa if abs(sa) > abs(sb) else sb
    scale = add(abs(sa), abs(sb))
    if scale == 0:
        return f(0), f(0), f(1), f(0)
    ra = div(sa, scale)
    rb = div(sb, scale)
    r = f(np.float64(scale) * math.sqrt(np.float64(add(mul(ra, ra), mul(rb, rb)))))
    r = r if roe >= 0 else -r
    c = div(sa, r)
    s = div(sb, r)
    z = f(1)
    if abs(sa) > abs(sb):
        z = s
    if abs(sb) >= abs(sa) and c != 0:
        z = div(1, c)
    return r, z, c, s


@njit
def rot(matrix, k, l, c, s):
    for i in range(3):
        x, y = matrix[i][k], matrix[i][l]
        matrix[i][k] = add(mul(c, x), mul(s, y))
        matrix[i][l] = sub(mul(c, y), mul(s, x))


@njit
def svdc_3(a):
    # Zero-index translation of ITK 5.3 v3p_netlib_ssvdc_(job=21).
    x = np.asarray(a, np.float32).copy()
    u = np.zeros((3, 3), np.float32)
    v = np.zeros((3, 3), np.float32)
    s = np.zeros(3, np.float32)
    e = np.zeros(3, np.float32)
    work = np.zeros(3, np.float32)
    for l in range(2):
        s[l] = norm(x[l:, l])
        if s[l] != 0:
            if x[l][l] != 0:
                s[l] = math.copysign(s[l], x[l][l])
            r = div(1, s[l])
            for i in range(l, 3):
                x[i][l] = mul(r, x[i][l])
            x[l][l] = add(x[l][l], 1)
        s[l] = -s[l]
        for j in range(l + 1, 3):
            if s[l] != 0:
                # sdot_ returns a double whose value was accumulated in float.
                t = f(-np.float64(dot(x[l:, l], x[l:, j])) / np.float64(x[l][l]))
                for i in range(l, 3):
                    x[i][j] = add(x[i][j], mul(t, x[i][l]))
            e[j] = x[l][j]
        for i in range(l, 3):
            u[i][l] = x[i][l]
        if l == 0:
            lp1 = l + 1
            e[l] = norm(e[lp1:])
            if e[l] != 0:
                if e[lp1] != 0:
                    e[l] = math.copysign(e[l], e[lp1])
                r = div(1, e[l])
                for i in range(lp1, 3):
                    e[i] = mul(r, e[i])
                e[lp1] = add(e[lp1], 1)
            e[l] = -e[l]
            if e[l] != 0:
                for i in range(lp1, 3):
                    work[i] = f(0)
                for j in range(lp1, 3):
                    for i in range(lp1, 3):
                        work[i] = add(work[i], mul(e[j], x[i][j]))
                for j in range(lp1, 3):
                    r = div(-e[j], e[lp1])
                    for i in range(lp1, 3):
                        x[i][j] = add(x[i][j], mul(r, work[i]))
            for i in range(lp1, 3):
                v[i][l] = e[i]
    s[2] = x[2][2]
    e[1] = x[1][2]
    e[2] = f(0)
    # Generate U in reverse l=1,0 order.
    for i in range(3):
        u[i][2] = f(0)
    u[2][2] = f(1)
    for l in (1, 0):
        if s[l] != 0:
            for j in range(l + 1, 3):
                t = f(-np.float64(dot(u[l:, l], u[l:, j])) / np.float64(u[l][l]))
                for i in range(l, 3):
                    u[i][j] = add(u[i][j], mul(t, u[i][l]))
            for i in range(l, 3):
                u[i][l] = mul(-1, u[i][l])
            u[l][l] = add(u[l][l], 1)
            for i in range(l):
                u[i][l] = f(0)
        else:
            for i in range(3):
                u[i][l] = f(0)
            u[l][l] = f(1)
    # Generate V in reverse l=2,1,0 order.
    for l in (2, 1, 0):
        if l == 0 and e[l] != 0:
            for j in range(l + 1, 3):
                t = f(-np.float64(dot(v[l + 1:, l], v[l + 1:, j])) / np.float64(v[l + 1][l]))
                for i in range(l + 1, 3):
                    v[i][j] = add(v[i][j], mul(t, v[i][l]))
        for i in range(3):
            v[i][l] = f(0)
        v[l][l] = f(1)
    m = 3
    mm = m
    it = 0
    while m:
        if it >= 1000:
            raise RuntimeError('ssvdc convergence')
        for ll in range(m):
            l = m - ll - 2
            if l < 0:
                break
            test = add(abs(s[l]), abs(s[l + 1]))
            ztest = add(test, abs(e[l]))
            if ztest == test:
                e[l] = f(0)
                break
        if l == m - 2:
            kase = 4
        else:
            for lls in range(l + 2, m + 2):
                ls = m - lls + l + 1
                if ls == l:
                    break
                test = f(0)
                if ls != m - 1:
                    test = add(test, abs(e[ls]))
                if ls != l + 1:
                    test = add(test, abs(e[ls - 1]))
                ztest = add(test, abs(s[ls]))
                if ztest == test:
                    s[ls] = f(0)
                    break
            if ls == l:
                kase = 3
            elif ls == m - 1:
                kase = 1
            else:
                kase = 2
                l = ls
        l += 1
        if kase == 1:
            fval = e[m - 2]
            e[m - 2] = f(0)
            for k in range(m - 2, l - 1, -1):
                t1, fval, cs, sn = rotg(s[k], fval)
                s[k] = t1
                if k != l:
                    fval = mul(-sn, e[k - 1])
                    e[k - 1] = mul(cs, e[k - 1])
                rot(v, k, m - 1, cs, sn)
        elif kase == 2:
            fval = e[l - 1]
            e[l - 1] = f(0)
            for k in range(l, m):
                t1, fval, cs, sn = rotg(s[k], fval)
                s[k] = t1
                fval = mul(-sn, e[k])
                e[k] = mul(cs, e[k])
                rot(u, k, l - 1, cs, sn)
        elif kase == 3:
            scale = max(abs(s[m - 1]), abs(s[m - 2]), abs(e[m - 2]),
                        abs(s[l]), abs(e[l]))
            sm = div(s[m - 1], scale)
            smm1 = div(s[m - 2], scale)
            emm1 = div(e[m - 2], scale)
            sl = div(s[l], scale)
            el = div(e[l], scale)
            b = div(add(mul(add(smm1, sm), sub(smm1, sm)), mul(emm1, emm1)), 2)
            c = mul(mul(sm, emm1), mul(sm, emm1))
            shift = f(0)
            if b != 0 or c != 0:
                shift = f(math.sqrt(np.float64(add(mul(b, b), c))))
                if b < 0:
                    shift = -shift
                shift = div(c, add(b, shift))
            fval = add(mul(add(sl, sm), sub(sl, sm)), shift)
            g = mul(sl, el)
            for k in range(l, m - 1):
                fval, g, cs, sn = rotg(fval, g)
                if k != l:
                    e[k - 1] = fval
                fval = add(mul(cs, s[k]), mul(sn, e[k]))
                e[k] = sub(mul(cs, e[k]), mul(sn, s[k]))
                g = mul(sn, s[k + 1])
                s[k + 1] = mul(cs, s[k + 1])
                rot(v, k, k + 1, cs, sn)
                fval, g, cs, sn = rotg(fval, g)
                s[k] = fval
                fval = add(mul(cs, e[k]), mul(sn, s[k + 1]))
                s[k + 1] = add(mul(-sn, e[k]), mul(cs, s[k + 1]))
                g = mul(sn, e[k + 1])
                e[k + 1] = mul(cs, e[k + 1])
                if k < 2:
                    rot(u, k, k + 1, cs, sn)
            e[m - 2] = fval
            it += 1
        else:
            if s[l] < 0:
                s[l] = -s[l]
                for i in range(3):
                    v[i][l] = mul(-1, v[i][l])
            while l != mm - 1 and s[l] < s[l + 1]:
                s[l], s[l + 1] = s[l + 1], s[l]
                if l < 2:
                    for i in range(3):
                        v[i][l], v[i][l + 1] = v[i][l + 1], v[i][l]
                        u[i][l], u[i][l + 1] = u[i][l + 1], u[i][l]
                l += 1
            it = 0
            m -= 1
    return u, s, v

@njit
def svd_inverse_3(a):
    u, w, v = svdc_3(a)
    wmin = f(1e-4 * np.float64(np.max(np.abs(w))))
    wi = np.zeros(3, np.float32)
    for k in range(3):
        if abs(w[k]) >= wmin:
            wi[k] = div(1, w[k])
    tmp = np.zeros((3, 3), np.float32)
    inverse = np.zeros((3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            val = f(0)
            for k in range(3):
                val = add(val, mul(wi[i] if i == k else f(0), u[j][k]))
            tmp[i][j] = val
    for i in range(3):
        for j in range(3):
            val = f(0)
            for k in range(3):
                val = add(val, mul(v[i][k], tmp[k][j]))
            inverse[i][j] = val
    return inverse
