import os.path as op

from os import makedirs
from math import log10
from argparse import ArgumentParser, RawTextHelpFormatter

def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 hmm.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Perform the specified algorithm, with given sequences and parameters.\n\n'
        '  Example syntax:\n'
        '    python3 hmm.py -vv viterbi seq.fasta A.tsv E.tsv\n'
        '    python3 hmm.py baumwelch in.fa priorA priorE -o ./outputs -i 1')

    # Positionals
    parser.add_argument('fasta', help='path to a FASTA formatted input file')
    parser.add_argument('transition', help='path to a TSV formatted transition matrix')
    parser.add_argument('emission', help='path to a TSV formatted emission matrix')

    # Optionals
    parser.add_argument('-v', '--verbose', dest='verbosity', action='count', default=0,
        help='print verbose output specific to the algorithm\n'
             '  (print even more output if flag is given twice)')

    parser.add_argument('-o', dest='out_dir',
        help='path to a directory where output files are saved\n'
             '  (directory will be made if it does not exist)\n'
             '  (file names and contents depend on algorithm)')

    return parser.parse_args()

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

def serialize(dictionary, sequence=False):
    keys = sorted(dictionary.keys())
    if sequence:      # Trellis
        keys = sort_states(keys)
        ix   = range(len(sequence)+2)
        out  = ['\t'.join(list(' -' + sequence + '-'))]
    elif 'B' in keys: # Transition matrix
        keys = sort_states(keys)
        ix   = keys
        out  = ['\t'.join([' ']+keys)]
    else:             # Emission matrix
        ix  = sorted(dictionary[keys[0]].keys())
        out = ['\t'.join([' ']+ix)]
    for k in keys:
        line = k + '\t' + '\t'.join(['%1.2e' % dictionary[k][i] for i in ix])
        out.append(line)
    return '\n'.join(out)

def sort_states(states):
    "Sort a list of states, while making sure 'B' and 'E' respectively start and end the list."
    Q = sorted(states)
    Q.remove('B')
    Q.remove('E')
    return ['B'] + Q + ['E']

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

def main(args = False):
    
    # Process arguments and load specified files
    if not args: args = parse_args()

    verbosity = args.verbosity
    set_X, labels = load_fasta(args.fasta)  # List of sequences, list of labels
    A = load_tsv(args.transition) # Nested Q -> Q dictionary
    E = load_tsv(args.emission)   # Nested Q -> S dictionary
    
    def save(filename, contents):
        if args.out_dir:
            makedirs(args.out_dir, exist_ok=True) # Make sure the output directory exists.
            path = op.join(args.out_dir,filename)
            with open(path,'w') as f: f.write(contents)
        # Note this function does nothing if no out_dir is specified!

    # VITERBI
    for j,X in enumerate(set_X):
        Q, P, T = viterbi(X,A,E)

        label = labels[j]
        save('%s.path' % label, Q)
        save('%s.matrix' % label, serialize(T,X))
        save('%s.p' % label, '%1.2e' % P)
        print('>%s\n Path = %s' % (label,Q))
        if verbosity: print(' Seq  = %s\n P    = %1.2e\n' % (X,P))
        if verbosity >= 2: print_trellis(T, X)

if __name__ == '__main__':
	main()