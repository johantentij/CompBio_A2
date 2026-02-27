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
    
    set_X = [
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
        print("sequence:", X)
        Q, P, T = viterbi(X,A,E)
        label = labels[j]
        print("most likely path:", Q)
        print("probability:", P)
        print("\n")

if __name__ == '__main__':
	main()