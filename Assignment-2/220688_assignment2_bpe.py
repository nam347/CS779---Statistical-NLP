import argparse
import heapq
from collections import Counter, defaultdict
import time

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def train_bpe_tokenizer(text, vocab_size):
    vocab = {i: bytes([i]) for i in range(256)}
    
    word_counts = Counter(text.strip().split())
    
    corpus = []
    for word, freq in word_counts.items():
        symbols = list(word.encode('utf-8'))
        corpus.append([symbols, freq])
    
    merges = {}
    
    pair_counts = defaultdict(int)
    pair_positions = defaultdict(list)
    
    for corpus_idx, (symbols, freq) in enumerate(corpus):
        for pos in range(len(symbols) - 1):
            pair = (symbols[pos], symbols[pos + 1])
            pair_counts[pair] += freq
            pair_positions[pair].append((corpus_idx, pos))
    
    # Used heap for efficient max extraction
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    
    for i in range(vocab_size - 256):
        if not heap:
            break
        
        while heap:
            neg_count, best_pair = heapq.heappop(heap)
            if pair_counts[best_pair] == -neg_count:
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
                old_left_pair = (symbols[pos - 1], symbols[pos])
                pair_counts[old_left_pair] -= freq
                affected_pairs.add(old_left_pair)
            
            if pos + 2 < len(symbols):
                old_right_pair = (symbols[pos + 1], symbols[pos + 2])
                pair_counts[old_right_pair] -= freq
                affected_pairs.add(old_right_pair)
            
            symbols[pos] = new_token_id
            del symbols[pos + 1]
            
            if pos > 0:
                new_left_pair = (symbols[pos - 1], symbols[pos])
                pair_counts[new_left_pair] += freq
                affected_pairs.add(new_left_pair)
                pair_positions[new_left_pair].append((corpus_idx, pos - 1))
            
            if pos + 1 < len(symbols):
                new_right_pair = (symbols[pos], symbols[pos + 1])
                pair_counts[new_right_pair] += freq
                affected_pairs.add(new_right_pair)
                pair_positions[new_right_pair].append((corpus_idx, pos))
        
        del pair_positions[best_pair]
        del pair_counts[best_pair]
        
        for pair in affected_pairs:
            if pair_counts[pair] > 0:
                heapq.heappush(heap, (-pair_counts[pair], pair))
    
    final_vocab_str = ["<pad>", "<unk>", "<s>", "</s>"]
    for i in sorted(vocab.keys()):
        final_vocab_str.append(vocab[i].decode('utf-8', errors='replace'))
    
    return final_vocab_str, merges, vocab

def tokenize(text, merges):

    merge_priority = {pair: i for i, pair in enumerate(merges.keys())}
    tokens = []

    words = text.strip().split()

    for wi, word in enumerate(words):
        symbols = list(word.encode("utf-8"))
        n = len(symbols)

        heap = []
        for i in range(n - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair in merge_priority:
                heapq.heappush(heap, (merge_priority[pair], i, pair))

        while heap:
            priority, pos, pair = heapq.heappop(heap)
            if pos >= len(symbols) - 1:
                continue
            if (symbols[pos], symbols[pos + 1]) != pair:
                continue 

            new_token = merges[pair]
            symbols[pos:pos + 2] = [new_token]

            if pos > 0:
                left_pair = (symbols[pos - 1], symbols[pos])
                if left_pair in merge_priority:
                    heapq.heappush(heap, (merge_priority[left_pair], pos - 1, left_pair))
            if pos < len(symbols) - 1:
                right_pair = (symbols[pos], symbols[pos + 1])
                if right_pair in merge_priority:
                    heapq.heappush(heap, (merge_priority[right_pair], pos, right_pair))

        tokens.extend(symbols)
        if wi != len(words) - 1:   
            tokens.append(32)    

    return tokens

def detokenize(tokens, token_map):
    words = []
    current_bytes = bytearray()
    
    for token_id in tokens:
        if token_id == 32: 
            if current_bytes:
                words.append(current_bytes.decode("utf-8", errors="replace"))
                current_bytes = bytearray()
            words.append(" ")  
        elif token_id in token_map:
            current_bytes.extend(token_map[token_id])
    
    if current_bytes:
        words.append(current_bytes.decode("utf-8", errors="replace"))
    
    return "".join(words)

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
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
            
            f.write(f"{token_id}\t{token_str}\n")

def save_tokens(tokens, rollno, token_map):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            if tok in token_map:
                human_readable = token_map[tok].decode("utf-8", errors="replace")
            elif tok == 32:
                human_readable = " "  
            else:
                human_readable = "<unk>"
            f.write(f"{tok}\t{human_readable}\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()
    
    rollno = "220688"
    
    train_text = load_training_data(args.train)
    final_vocab_str, merges, token_map = train_bpe_tokenizer(train_text, args.vocab_size)

    save_vocab(token_map, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, merges)
    save_tokens(tokens, rollno, token_map)
    
    detok_text = detokenize(tokens, token_map)
    save_detokenized(detok_text, rollno)
    
    total_time = time.time() - start_time
    print(f"Training complete. Final vocab size {len(final_vocab_str)}. Time: {total_time:.2f}s")
