import argparse
import math
from collections import Counter, defaultdict
import heapq
import time

def compute_delta_L(pair, pair_counts, token_freqs, N):
    a, b = pair
    f_new = pair_counts.get(pair, 0)
    if f_new <= 0 or N <= 0:
        return float("-inf")
    f_a, f_b = token_freqs.get(a, 0), token_freqs.get(b, 0)
    ratio = (f_new ) / (N )
    if ratio <= 0:
        return float("-inf")
    return (f_new - (f_a + f_b)) * math.log(ratio)

def build_word_symbols(text):
    """Build word-to-symbols mapping and initial token frequencies"""
    word_counts = Counter(text.split())
    if "" in word_counts:
        del word_counts[""]
    
    word_to_symbols = {}
    word_to_freq = {}
    token_freqs = Counter()
    
    for word_id, (word, freq) in enumerate(word_counts.items()):
        symbols = tuple(list(word))  
        word_to_symbols[word_id] = symbols
        word_to_freq[word_id] = freq
        
        for symbol in symbols:
            token_freqs[symbol] += freq
    
    return word_to_symbols, word_to_freq, token_freqs

def build_pair_data(word_to_symbols, word_to_freq):
    """Build pair counts and position tracking"""
    pair_counts = Counter()
    pair_positions = defaultdict(set)
    
    for word_id, symbols in word_to_symbols.items():
        freq = word_to_freq[word_id]
        for pos in range(len(symbols) - 1):
            pair = (symbols[pos], symbols[pos + 1])
            pair_counts[pair] += freq
            pair_positions[pair].add((word_id, pos))
    
    return pair_counts, pair_positions

def apply_merge(word_to_symbols, word_to_freq, best_pair, new_token, 
                pair_counts, pair_positions, token_freqs):
    """Apply merge operation and update all data structures"""
    a, b = best_pair
    affected_pairs = set()
    total_merged_freq = 0
    
    occurrences = list(pair_positions.get(best_pair, set()))
    if not occurrences:
        return affected_pairs, total_merged_freq
    
    occs_by_word = defaultdict(list)
    for word_id, pos in occurrences:
        occs_by_word[word_id].append(pos)
    
    for word_id, pos_list in occs_by_word.items():
        pos_list_sorted = sorted(set(pos_list), reverse=True)
        symbols = list(word_to_symbols[word_id])
        freq = word_to_freq[word_id]
        
        for pos in pos_list_sorted:
            if pos < 0 or pos >= len(symbols) - 1:
                continue
            if not (symbols[pos] == a and symbols[pos + 1] == b):
                continue
            
            total_merged_freq += freq
            
            if pos > 0:
                left_pair = (symbols[pos - 1], symbols[pos])
                pair_counts[left_pair] -= freq
                pair_positions[left_pair].discard((word_id, pos - 1))
                if pair_counts[left_pair] <= 0:
                    pair_counts.pop(left_pair, None)
                affected_pairs.add(left_pair)
            
            if pos + 2 < len(symbols):
                right_pair = (symbols[pos + 1], symbols[pos + 2])
                pair_counts[right_pair] -= freq
                pair_positions[right_pair].discard((word_id, pos + 1))
                if pair_counts[right_pair] <= 0:
                    pair_counts.pop(right_pair, None)
                affected_pairs.add(right_pair)
            
            pair_counts[best_pair] -= freq
            pair_positions[best_pair].discard((word_id, pos))
            
            token_freqs[a] -= freq
            if token_freqs[a] <= 0:
                del token_freqs[a]
            token_freqs[b] -= freq
            if token_freqs[b] <= 0:
                del token_freqs[b]
            token_freqs[new_token] = token_freqs.get(new_token, 0) + freq
            
            symbols[pos] = new_token
            del symbols[pos + 1] 
            
            if pos > 0:
                new_left_pair = (symbols[pos - 1], symbols[pos])
                pair_counts[new_left_pair] = pair_counts.get(new_left_pair, 0) + freq
                pair_positions[new_left_pair].add((word_id, pos - 1))
                affected_pairs.add(new_left_pair)
            
            if pos + 1 < len(symbols):
                new_right_pair = (symbols[pos], symbols[pos + 1])
                pair_counts[new_right_pair] = pair_counts.get(new_right_pair, 0) + freq
                pair_positions[new_right_pair].add((word_id, pos))
                affected_pairs.add(new_right_pair)
        
        word_to_symbols[word_id] = tuple(symbols)
    
    if pair_counts.get(best_pair, 0) <= 0:
        pair_counts.pop(best_pair, None)
    pair_positions.pop(best_pair, None)
    
    return affected_pairs, total_merged_freq


def train_wordpiece_tokenizer(text, vocab_size, verbose=True):
    t0 = time.time()
    log_interval_seconds = 30.0
    last_log = t0

    
    word_to_symbols, word_to_freq, token_freqs = build_word_symbols(text)
    pair_counts, pair_positions = build_pair_data(word_to_symbols, word_to_freq)
    
    N = sum(token_freqs.values())
    
    specials = ["<pad>", "<unk>", "<s>", "</s>"]
    base_tokens = sorted(token_freqs.keys())
    continuation_tokens = ["##" + t for t in base_tokens]
    
    vocab = specials + base_tokens + continuation_tokens
    vocab_set = set(vocab)
    
    heap = []
    for pair, f_new in pair_counts.items():
        a, b = pair
        f_a = token_freqs.get(a, 0)
        f_b = token_freqs.get(b, 0)
        delta = compute_delta_L(pair, pair_counts, token_freqs, N)
        if delta > float("-inf") and delta > 0:
            heapq.heappush(heap, (-delta, pair))
    
    merges_done = 0
    target_vocab_size = vocab_size
    
    while len(vocab) < target_vocab_size and heap:
        best_pair = None
        
        while heap and best_pair is None:
            neg_delta, pair = heapq.heappop(heap)
            f_new = pair_counts.get(pair, 0)
            
            if f_new <= 0:
                continue
                
            a, b = pair
            current_delta = compute_delta_L(pair, pair_counts, token_freqs, N)
            
            if current_delta > 1e-12:
                best_pair = pair
                break
        
        if best_pair is None:
            break
        
        a, b = best_pair
        new_token = a + b
        
        additions = []
        if new_token not in vocab_set:
            additions.append(new_token)
        continuation_token = "##" + new_token
        if continuation_token not in vocab_set and len(new_token) > 1:
            additions.append(continuation_token)
        
        for tok in additions:
            if len(vocab) < target_vocab_size:
                vocab.append(tok)
                vocab_set.add(tok)
            else:
                break
        

        affected_pairs, total_merged_freq = apply_merge(
            word_to_symbols, word_to_freq, best_pair, new_token,
            pair_counts, pair_positions, token_freqs
        )
        
        merges_done += 1
        
        N = N - total_merged_freq
        if N < 0:
            N = 0
        
        for pair in affected_pairs:
            f_new = pair_counts.get(pair, 0)
            if f_new <= 0:
                continue
            a_p, b_p = pair
            f_a = token_freqs.get(a_p, 0)
            f_b = token_freqs.get(b_p, 0)
            delta = compute_delta_L(pair, pair_counts, token_freqs, N)
            if delta > float("-inf") and delta > 0:
                heapq.heappush(heap, (-delta, pair))
        
        now = time.time()
        if verbose and (now - last_log >= log_interval_seconds or merges_done % 250 == 0):
            elapsed = now - t0
            if verbose:
                print(f"[merges={merges_done}] vocab_size={len(vocab)} elapsed={elapsed:.1f}s heap_candidates={len(heap)}")
            last_log = now
    
    total_elapsed = time.time() - t0
    if verbose:
        print(f"Training complete. Performed merges={merges_done}; final vocab size={len(vocab)}; time={total_elapsed:.1f}s")
    
    tokenizer = {"vocab_set": vocab_set, "reserved": specials}
    return vocab, tokenizer


def tokenize(text, tokenizer):
    vocab = tokenizer["vocab_set"]
    tokens = []
    
    for word in text.split():
        if word == "":
            continue
            
        if word in vocab:
            tokens.append(word)
            continue
        
        i = 0
        word_tokens = []
        while i < len(word):
            match_found = False
            for j in range(len(word), i, -1):
                subword = word[i:j]
                token_candidate = subword if i == 0 else "##" + subword
                
                if token_candidate in vocab:
                    word_tokens.append(token_candidate)
                    i = j
                    match_found = True
                    break
            
            if not match_found:
                word_tokens.append("<unk>")
                break    
        tokens.extend(word_tokens)
    
    return tokens

def detokenize(tokens, tokenizer):
    reserved = tokenizer.get("reserved", [])
    words = []
    current_word = ""
    
    for tok in tokens:
        if tok in reserved:
            continue
        elif tok == "<unk>":
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append("<unk>")
        elif tok.startswith("##"):
            current_word += tok[2:]
        else:
            if current_word:
                words.append(current_word)
            current_word = tok
    
    if current_word:
        words.append(current_word)
    
    return " ".join(words)



def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_wp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(tok + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_wp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

import matplotlib.pyplot as plt
import math
from collections import Counter

def compute_nll(word_to_symbols, word_to_freq, token_freqs):
    """
    Compute negative log-likelihood (NLL) of corpus given current tokenization.
    """
    N = sum(token_freqs.values())
    nll = 0.0
    for word_id, symbols in word_to_symbols.items():
        freq = word_to_freq[word_id]
        for symbol in symbols:
            f = token_freqs.get(symbol, 1)  # avoid zero
            prob = f / N
            nll -= freq * math.log(prob)
    return nll



if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "220688"
    
    with open(args.train, "r", encoding="utf-8") as f:
        train_text = f.read()

    # Train tokenizer
    vocab, tokenizer = train_wordpiece_tokenizer(train_text, args.vocab_size, verbose=True)
    save_vocab(vocab, rollno, args.vocab_size)

    # Load input data and tokenize
    with open(args.input, "r", encoding="utf-8") as f:
        input_text = f.read()

    tokens = tokenize(input_text, tokenizer)
    save_tokens(tokens, rollno)

    # Detokenize and save
    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)

    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of tokens generated: {len(tokens)}")
