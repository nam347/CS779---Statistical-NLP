import argparse
import heapq
from collections import defaultdict, Counter
import unicodedata


SPACE_SYMBOL = "▁"

def normalize_text(text):
    return unicodedata.normalize("NFKC", text)

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = normalize_text(text)
    text = text.replace("\n", f"{SPACE_SYMBOL}\n")
    text = text.replace(" ", SPACE_SYMBOL)
    return text

def train_sp_tokenizer(text, vocab_size):
    word_counts = Counter(text.split(SPACE_SYMBOL))
    
    corpus = []
    for word, freq in word_counts.items():
        symbols = list(word.encode('utf-8'))
        corpus.append([symbols, freq])
    
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    
    pair_counts = defaultdict(int)
    pair_positions = defaultdict(list)
    for idx, (symbols, freq) in enumerate(corpus):
        for pos in range(len(symbols) - 1):
            pair = (symbols[pos], symbols[pos + 1])
            pair_counts[pair] += freq
            pair_positions[pair].append((idx, pos))
    
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    
    for i in range(vocab_size - 256):
        if not heap:
            break
        while heap:
            neg_count, best_pair = heapq.heappop(heap)
            if pair_counts.get(best_pair, 0) == -neg_count:
                break
        else:
            break
        
        new_token_id = 256 + i
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_token_id
        
        affected_pairs = set()
        for corpus_idx, pos in pair_positions[best_pair]:
            symbols, freq = corpus[corpus_idx]
            if pos >= len(symbols) - 1 or (symbols[pos], symbols[pos + 1]) != best_pair:
                continue
            if pos > 0:
                left_pair = (symbols[pos - 1], symbols[pos])
                pair_counts[left_pair] -= freq
                affected_pairs.add(left_pair)
            if pos + 2 < len(symbols):
                right_pair = (symbols[pos + 1], symbols[pos + 2])
                pair_counts[right_pair] -= freq
                affected_pairs.add(right_pair)
            
            symbols[pos] = new_token_id
            del symbols[pos + 1]
            
            if pos > 0:
                new_left = (symbols[pos - 1], symbols[pos])
                pair_counts[new_left] += freq
                affected_pairs.add(new_left)
                pair_positions[new_left].append((corpus_idx, pos - 1))
            if pos < len(symbols) - 1:
                new_right = (symbols[pos], symbols[pos + 1])
                pair_counts[new_right] += freq
                affected_pairs.add(new_right)
                pair_positions[new_right].append((corpus_idx, pos))
        
        pair_counts.pop(best_pair, None)
        pair_positions.pop(best_pair, None)
        
        for pair in affected_pairs:
            if pair_counts[pair] > 0:
                heapq.heappush(heap, (-pair_counts[pair], pair))
    
    return vocab, merges

def tokenize(text, merges):
    text = normalize_text(text).replace(" ", SPACE_SYMBOL)
    symbols = list(text.encode("utf-8"))
    
    merge_priority = {pair: i for i, pair in enumerate(merges.keys())}
    
    heap = []
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i+1])
        if pair in merge_priority:
            heapq.heappush(heap, (merge_priority[pair], i, pair))
    
    while heap:
        priority, pos, pair = heapq.heappop(heap)
        if pos >= len(symbols) - 1:
            continue
        if (symbols[pos], symbols[pos + 1]) != pair:
            continue
        
        new_token = merges[pair]
        symbols[pos:pos+2] = [new_token]
        
        if pos > 0:
            left_pair = (symbols[pos-1], symbols[pos])
            if left_pair in merge_priority:
                heapq.heappush(heap, (merge_priority[left_pair], pos-1, left_pair))
        if pos < len(symbols) - 1:
            right_pair = (symbols[pos], symbols[pos+1])
            if right_pair in merge_priority:
                heapq.heappush(heap, (merge_priority[right_pair], pos, right_pair))
    
    return symbols

def detokenize(tokens, vocab):
    b = bytearray()
    for t in tokens:
        token_bytes = vocab.get(t)
        if token_bytes is None:
            token_bytes = b"?"
        b.extend(token_bytes)
    text = b.decode("utf-8", errors="replace")
    return text.replace(SPACE_SYMBOL, " ")


def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(f"{tok}\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token_id in sorted(vocab.keys()):
            token_bytes = vocab[token_id]
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                token_str = token_bytes.hex()

            if token_str == " ":
                token_str = "<space>"
            elif token_str == "\n":
                token_str = "<newline>"
            elif token_str == "\t":
                token_str = "<tab>"
            elif token_str == SPACE_SYMBOL:
                token_str = "<whitespace_marker>"

            f.write(f"{token_id}\t{token_str}\n")


import time
if __name__ == "__main__":
    start_time= time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()
    
    rollno = "220688"
    
    train_text = load_training_data(args.train)
    vocab, merges = train_sp_tokenizer(train_text, args.vocab_size)
    
    with open(args.input, "r", encoding="utf-8") as f:
        input_text = f.read()
    save_vocab(vocab, rollno, args.vocab_size)
    
    tokens = tokenize(input_text, merges)
    save_tokens(tokens, rollno)
    
    detok_text = detokenize(tokens, vocab)
    save_detokenized(detok_text, rollno)
    
    total_time = time.time() - start_time
    print(f"Training complete. Time: {total_time:.2f}s")