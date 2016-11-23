import numpy as np


def preprocess(sequences):
    # build word-to-index dictionary 
    word_to_idx = {'<START>': 0, '<END>': 1, '<PAD>': 2, '<UNK>': 3}
    i = 4
    max_length = 0
    for seq in sequences:
        words = seq.replace('\n', '').lower().split(' ')

        if max_length < len(words):
            max_length = len(words)

        for word in words:
            if not word in word_to_idx:
                word_to_idx[word] = i
                i += 1
        
        # set sequence length to max length (short sequences will be padded) and +2 for '<START>' and '<END>'
        seq_length = max_length + 2
        vocab_size = len(word_to_idx)
        
    # build sequence vectors
    seq_with_id = np.ndarray(shape=(len(sequences), seq_length), dtype=np.int64) 
    for i, seq in enumerate(sequences):
        words = seq.replace('\n', '').lower().split(' ')

        for j in range(seq_length):
            if j == 0:
                seq_with_id[i, j] = word_to_idx['<START>']
            elif j-1 < len(words):
                seq_with_id[i, j] = word_to_idx[words[j-1]]
            elif j-1 == len(words):
                seq_with_id[i, j] = word_to_idx['<END>']
            else:
                seq_with_id[i, j] = word_to_idx['<PAD>']       

    # build mask 
    mask = (seq_with_id != word_to_idx['<PAD>'])
    
    return seq_with_id, mask, word_to_idx, seq_length, vocab_size


def decode_sequence(sequences, word_to_idx):
    # build index-to-word dictionary for decoding sequence vectors
    idx_to_word = {i:w for w,i in word_to_idx.iteritems()}
    
    assert len(sequences.shape) == 2
    batch_size, seq_length = sequences.shape
        
    decoded = []
    for seq in sequences:
        d = []
        for idx in seq:
            word = idx_to_word[idx]
            if word == '<START>':
                continue
            elif word == '<END>':
                d.append('.')
                break 
            else:
                d.append(word)
        decoded.append(' '.join(d))
    
    return decoded