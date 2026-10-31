"""Generate the circuit task's semantic tests with an independent, sequential reference solver.

Usage: python gen_correctness.py
"""
from collections import deque
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
MASK = (1 << 64) - 1
TYPES = ('AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'NOT', 'MUX')


def mix(x):
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def circuit_reference(F, gates, d, out, M, T, seed):
    """One scalar stimulus at a time; Kahn order; snapshot FF updates."""
    bg = 66 + F
    successors = [[] for _ in gates]
    degree = [0] * len(gates)
    for k, (_, inputs) in enumerate(gates):
        for s in inputs:
            if s >= bg:
                successors[s - bg].append(k)
                degree[k] += 1
    queue = deque(k for k, deg in enumerate(degree) if not deg)
    order = []
    while queue:
        k = queue.popleft()
        order.append(k)
        for j in successors[k]:
            degree[j] -= 1
            if not degree[j]:
                queue.append(j)
    assert len(order) == len(gates), 'cyclic circuit'
    result = []
    for m in range(M):
        v = [0, 1] + [0] * (64 + F + len(gates))
        state, digest = mix(seed ^ m), 0
        for t in range(T):
            state = (state + 0x9E3779B97F4A7C15) & MASK
            r = mix((state - 0x9E3779B97F4A7C15) & MASK)
            for i in range(64):
                v[2 + i] = (r >> i) & 1
            for k in order:
                kind, inputs = gates[k]
                a = [v[s] for s in inputs]
                if kind == 0: value = a[0] & a[1]
                elif kind == 1: value = a[0] | a[1]
                elif kind == 2: value = a[0] ^ a[1]
                elif kind == 3: value = 1 ^ (a[0] & a[1])
                elif kind == 4: value = 1 ^ (a[0] | a[1])
                elif kind == 5: value = 1 ^ a[0] ^ a[1]
                elif kind == 6: value = 1 ^ a[0]
                else: value = a[2] if a[0] else a[1]
                v[bg + k] = value
            for j, s in enumerate(out):
                if v[s]: digest ^= mix(t * len(out) + j)
            snapshot = [v[s] for s in d]
            v[66:bg] = snapshot
        result.append(digest)
    return result


def write_circuit(name, F, gates, d, out, M, T, seed, reference=None):
    base = ROOT / '通用数字电路仿真器加速' / 'correctness'
    (base / 'inputs').mkdir(parents=True, exist_ok=True)
    (base / 'expected').mkdir(exist_ok=True)
    assert len(d) == F and 1 <= F <= 8192 and 1 <= len(out) <= 64
    lines = [f'64 {F} {len(gates)} {len(out)} {M} {T} {seed}', ' '.join(map(str, d))]
    lines += [TYPES[kind] + ' ' + ' '.join(map(str, inputs)) for kind, inputs in gates]
    lines.append(' '.join(map(str, out)))
    (base / 'inputs' / f'{name}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    result = circuit_reference(F, gates, d, out, M, T, seed) if reference is None else reference
    assert len(result) == M
    (base / 'expected' / f'{name}.txt').write_text(''.join(f'{h:016x}\n' for h in result), encoding='utf-8', newline='\n')


def circuits():
    write_circuit('01_ff_forward', 6, [], [1, 66, 67, 68, 69, 70], list(range(66, 72)), 3, 10, 0)
    write_circuit('02_ff_reverse', 6, [], [67, 68, 69, 70, 71, 1], list(range(66, 72)), 5, 10, MASK)
    # Direct PI/constant/FF outputs, direct PI/constant/self FF inputs, dead gates.
    write_circuit('03_direct_sources', 4, [(2, [2, 65]), (7, [3, 0, 1])],
                  [2, 1, 0, 69], [0, 1, 2, 65, 66, 67, 68, 69, 70, 71], 65, 17, MASK - 1)
    for case, (F, G, M, T) in enumerate([(7, 33, 17, 19), (19, 157, 257, 13), (9, 96, 3, 31)], 4):
        rng = random.Random(2026102500 + case)
        bg = 66 + F
        gates = []
        for k in range(G):
            kind = k % 8
            inputs = [rng.randrange(bg + k) for _ in range(1 if kind == 6 else 3 if kind == 7 else 2)]
            if len(inputs) >= 2 and k % 5 == 0: inputs[1] = inputs[0]
            gates.append((kind, inputs))
        perm = list(range(G))
        rng.shuffle(perm)
        remap = lambda s: s if s < bg else bg + perm[s - bg]
        shuffled = [None] * G
        for k, (kind, inputs) in enumerate(gates):
            shuffled[perm[k]] = (kind, [remap(s) for s in inputs])
        d = [remap(rng.randrange(bg + G)) for _ in range(F)]
        d[:4] = [1, 66, 2, 65]
        out = [0, 1, 2, 65, 66] + [remap(bg + k) for k in range(G - 12, G)]
        write_circuit(f'{case:02}_shuffled_dag', F, shuffled, d, out, M, T, rng.getrandbits(64))
    # Also exercise the full-size path, so a separate small-input implementation
    # cannot hide an in-place FF update in the optimized large-input path.
    # The dummy gates are NOT(0); none is observed.  Four FFs form two delays
    # in opposite index directions. Their waveform is known independently.
    digest = 0
    for t in range(512):
        active = [4]  # output 4 is the constant 1
        if t >= 1: active += [0, 3]
        if t >= 2: active += [1, 2]
        for j in active: digest ^= mix(t * 64 + j)
    write_circuit('07_fullsize_ff_snapshot', 8192, [(6, [0])] * 500000,
                  [1, 66, 69, 1] + [0] * 8188, [66, 67, 68, 69, 1] + [0] * 59,
                  4096, 512, 20261025, [digest] * 4096)


if __name__ == '__main__':
    circuits()
    print('Correctness inputs and reference outputs generated.')
