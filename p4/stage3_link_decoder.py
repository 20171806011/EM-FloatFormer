"""CPU sum-product soft decoder and ordered-statistics reprocessing of the native (128,64) code."""
from itertools import combinations

import numpy as np
from numba import njit
import torch


@njit
def _bp(channel, edge_bits, offsets, iterations):
    """Positive project LLR means one; check messages are computed in the positive-for-zero convention."""
    result = np.empty_like(channel)
    for w in range(channel.shape[0]):
        intrinsic = -np.clip(channel[w], -40.0, 40.0)
        post = intrinsic.copy()
        messages = np.zeros(len(edge_bits))
        v2c = intrinsic[edge_bits].copy()
        for iteration in range(iterations):
            post[:] = intrinsic
            for check in range(len(offsets)-1):
                start, end = offsets[check:check+2]
                product = 1.0
                zeroes = 0
                for e in range(start, end):
                    value = np.tanh(np.maximum(-35.0, np.minimum(35.0, v2c[e])) * 0.5)
                    messages[e] = value
                    if abs(value) > 1e-14:
                        product *= value
                    else:
                        zeroes += 1
                for e in range(start, end):
                    val = messages[e]
                    if zeroes == 0:
                        prod = product / val
                    elif zeroes == 1 and abs(val) <= 1e-14:
                        prod = product
                    else:
                        prod = 0.0
                    prod = max(-1+1e-14, min(1-1e-14, prod))
                    messages[e] = np.log1p(prod) - np.log1p(-prod)
                    post[edge_bits[e]] += messages[e]
            valid = True
            for check in range(len(offsets)-1):
                parity = 0
                for e in range(offsets[check], offsets[check+1]):
                    parity ^= int(post[edge_bits[e]] < 0)
                if parity:
                    valid = False
                    break
            if valid:
                break
            v2c = post[edge_bits] - messages
        result[w] = -post
    return result


@njit
def _get(row, col):
    return (row[col//64] >> np.uint64(col%64)) & np.uint64(1)


@njit
def _cost(c0, c1, table):
    value = 0.
    for j in range(8):
        value += table[j, int((c0 >> np.uint64(8*j)) & np.uint64(255))]
        value += table[j+8, int((c1 >> np.uint64(8*j)) & np.uint64(255))]
    return value


@njit
def _search(post, tables, gen, triples, quads):
    """Build a most-reliable independent basis (MRB), enumerate fixed test error patterns."""
    out = np.zeros((len(post),64),dtype=np.int64)
    for w in range(len(post)):
        rows = gen.copy()
        order = np.argsort(-np.abs(post[w]))
        c0 = np.uint64(0)
        c1 = np.uint64(0)
        pivots = np.empty(64, dtype=np.int64)
        k = 0
        for col in order:
            pivot = k
            while pivot < 64 and _get(rows[pivot],col)==0:
                pivot += 1
            if pivot == 64:
                continue
            if pivot != k:
                a,b=rows[k,0],rows[k,1]
                rows[k,0],rows[k,1]=rows[pivot,0],rows[pivot,1]
                rows[pivot,0],rows[pivot,1]=a,b
            for r in range(64):
                if r!=k and _get(rows[r],col):
                    rows[r,0] ^= rows[k,0]
                    rows[r,1] ^= rows[k,1]
            pivots[k] = col
            k += 1
            if k==64:
                break
        for i in range(64):
            if post[w,pivots[i]]>0:
                c0 ^= rows[i,0]
                c1 ^= rows[i,1]
        best = c0
        value = _cost(c0,c1,tables[w])
        for i in range(64):
            a,b = c0^rows[i,0],c1^rows[i,1]
            trial = _cost(a,b,tables[w])
            if trial<value:
                best,value = a,trial
            for j in range(i):
                aa,bb = a^rows[j,0],b^rows[j,1]
                trial = _cost(aa,bb,tables[w])
                if trial<value:
                    best,value=aa,trial
        for group in triples:
            i,j,k = group
            a = c0^rows[i,0]^rows[j,0]^rows[k,0]
            b = c1^rows[i,1]^rows[j,1]^rows[k,1]
            trial = _cost(a,b,tables[w])
            if trial<value:
                best,value = a,trial
        for group in quads:
            i,j,k,l = group
            a = c0^rows[i,0]^rows[j,0]^rows[k,0]^rows[l,0]
            b = c1^rows[i,1]^rows[j,1]^rows[k,1]^rows[l,1]
            trial = _cost(a,b,tables[w])
            if trial<value:
                best,value = a,trial
        for i in range(64):
            out[w,i] = (best >> np.uint64(i)) & np.uint64(1)
    return out


class SoftListDecoder:
    def __init__(self, code):
        if (code.k,code.n)!=(64,128):
            raise ValueError("This list reprocessor is for the native (128,64) LDPC code.")
        h = code.h_matrix.numpy()
        self.edge_bits = np.nonzero(h)[1].astype(np.int64)
        self.offsets = np.concatenate(([0],np.cumsum(h.sum(1)))).astype(np.int64)
        self.gen = np.ascontiguousarray(np.packbits(code.g_matrix.numpy().astype(np.uint8),
                                                   axis=1,bitorder="little")).view("<u8")
        self.triples = np.asarray(list(combinations(range(48,64),3)), dtype=np.int64)
        self.quads = np.asarray(list(combinations(range(56,64),4)), dtype=np.int64)

    def posterior(self, log_odds, iterations=20):
        data = np.ascontiguousarray(log_odds.reshape(-1,128).double().numpy())
        return torch.from_numpy(_bp(data,self.edge_bits,self.offsets,iterations)).reshape_as(log_odds)

    def list_decode(self, posterior, log_metric):
        # Symbol index i = sum_j bit_j * 2**j, in the same little-endian order as the code rows.
        metric = -log_metric.reshape(-1,16,2,16).float().numpy()
        table = metric[:,:,0,None,:] + metric[:,:,1,:,None]
        data = np.ascontiguousarray(posterior.reshape(-1,128).double().numpy())
        words = _search(data,np.ascontiguousarray(table.reshape(-1,16,256)),self.gen,self.triples,self.quads)
        return torch.from_numpy(words).reshape(*posterior.shape[:-1],64)
