import os.path as op

from os import makedirs
from math import log10
from argparse import ArgumentParser, RawTextHelpFormatter

def load_fasta(path):
    """Load a FASTA formatted set of sequences. Returns two lists: sequences and labels.
    Warning: Will likely throw errors if the file is not FASTA formatted!"""
    labs = []
    seqs = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                labs.append(line.strip()[1:])
                seqs.append('')
            else:
                seqs[-1] += line.strip()
    return seqs, labs

def load_tsv(path):
    "Load a TSV formatted set of (prior) parameters. Return as a nested dictionary."
    out = {}
    with open(path) as f:
        header = f.readline().strip().split('\t') # Read, strip and split the header line
        for line in f:
            ls = line.rstrip().split('\t')
            out[ls[0]] = {header[i]:float(v) for i,v in enumerate(ls[1:])}
    return out

def print_trellis(T,sequence):
    "Pretty print function for a Viterbi/Forward/Backward dynamic programming matrix."
    Q = sort_states(T.keys())
    X = '-' + sequence + '-'
    print('   '+''.join(['%-8s ' % s for s in X]))
    for q in Q:
        print('%2s ' % q + ''.join(['%1.2e ' % p for p in T[q]]))
    print('')

def print_params(A,E):
    "Pretty print function for the Transition matrix (from a nested dictionary)."
    QA = sort_states(A.keys())
    print('\n[A]   ' + ''.join('%-5s ' % j for j in QA))
    for i in QA:
        print('%5s ' % i + ''.join('%0.3f ' % A[i][j] for j in QA))
    QE = sorted(E.keys())
    S = sorted(E[QE[0]].keys())
    print('\n[E]   ' + ''.join('%-5s ' % s for s in S))
    for i in QE:
        print('%5s ' % i + ''.join('%0.3f ' % E[i][s] for s in S))
    print('')

def sort_states(states):
    "Sort a list of states, while making sure 'B' and 'E' respectively start and end the list."
    Q = sorted(states)
    Q.remove('B')
    return ['B'] + Q

def viterbi(X,A,E):
    """Given a single sequence, with Transition and Emission probabilities,
    return the most probable state path, the corresponding P(X), and trellis."""

    allStates = A.keys()
    emittingStates = E.keys()
    L = len(X) + 1

    # Initialize
    V = {k:[0] * L for k in allStates}
    V['B'][0] = 1.

    # Middle columns
    for i,s in enumerate(X):
        for l in emittingStates:
            terms = [V[k][i] * A[k][l] for k in allStates]
            V[l][i+1] = max(terms) * E[l][s]

    best_state = max(emittingStates, key=lambda state: V[state][len(X)])
    P = V[best_state][len(X)]

    pi = best_state
    l = best_state
    for i in range(len(X)-1,0,-1): # iterating backwards through the sequence
        for k in emittingStates:
            if abs(V[k][i] * A[k][l] * E[l][X[i]] - V[l][i+1]) < 1e-10:
                pi = k + pi
                l = k
                break

    return(pi,P,V) # Return the state path, Viterbi probability, and Viterbi trellis

def main():
    
    set_X = sequences = [
        "AGCGC",
        "AUUAU"
    ]
    labels = ["seq1", "seq2"]

    A = {
        'B': {'B': 0.0, 'E': 0.5, 'I': 0.5},
        'E': {'B': 0.0, 'E': 0.9, 'I': 0.1},
        'I': {'B': 0.0, 'E': 0.2, 'I': 0.8}
    }
    E = {
        'E': {'A': 0.25, 'U': 0.25, 'G': 0.25, 'C': 0.25},
        'I': {'A': 0.4, 'U': 0.4, 'G': 0.05, 'C': 0.15}
    }

    # VITERBI
    for j,X in enumerate(set_X):
        Q, P, T = viterbi(X,A,E)
        label = labels[j]
        print(Q)
        print(P)

if __name__ == '__main__':
	main()