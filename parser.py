import collections
import pickle

from chargetter import chargetter





def compute_ngram_frequencies(stream, n):

    frequencies = collections.defaultdict(int)

    current_buffer = collections.deque(maxlen=n)

    while len(current_buffer) < n-1:
        current_buffer.append(next(chargen))

    i = 0
    for next_elem in chargen:
        i += 1
        current_buffer.append(next_elem)
        n_gram = ''.join(current_buffer)
        frequencies[n_gram] += 1

    return frequencies




if __name__ == '__main__':
    chargen = chargetter(filename='War-peace.txt')

    n = 3
    frequencies = compute_ngram_frequencies(chargen, n)

    with open (f'{n}-gram-frequencies.txt', 'wb') as f:
        pickle.dump(dict(frequencies), f)``