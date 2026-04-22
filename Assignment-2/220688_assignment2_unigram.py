import argparse
from collections import Counter, defaultdict
import math
import numpy as np


def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def build_seed_vocab(text, max_size=20000, min_freq=2, max_sub_len=10):
    
    vocab = set(text)  
    counts = Counter()
    n = len(text)
    for L in range(2, max_sub_len+1):
        for i in range(n - L + 1):
            sub = text[i:i+L]
            counts[sub] += 1

    for sub, c in counts.most_common():
        if c >= min_freq:
            vocab.add(sub)
        if len(vocab) >= max_size:
            break
    
    return list(vocab)

def word_likelihood(word, probs, vocab_set):
   
    n = len(word)
    dp = np.full(n+1, -np.inf) 
    dp[0] = 0.0

    for i in range(1, n+1):
        for j in range(max(0, i-10), i):  
            sub = word[j:i]
            if sub in vocab_set:
                dp[i] = np.logaddexp(dp[i], dp[j] + math.log(probs[sub]))

    return dp[n]

def corpus_likelihood(corpus_words, probs, vocab_set):
    return sum(word_likelihood(w, probs, vocab_set) for w in corpus_words if w)



def train_unigram_tokenizer(text, vocab_size):
    words = text.split()

    vocab = build_seed_vocab(text, max_size=20000)
    vocab_set = set(vocab)

    probs = {tok: 1.0 / len(vocab) for tok in vocab}

    token_to_words = defaultdict(set)
    for idx, w in enumerate(words):
        n = len(w)
        for L in range(1, 11):
            for i in range(n - L + 1):
                sub = w[i:i + L]
                if sub in vocab_set:
                    token_to_words[sub].add(idx)

    while len(vocab) > vocab_size:
        losses = {}

        for tok in vocab:
            if len(tok) == 1:  
                continue

            affected = token_to_words.get(tok, set())
            if not affected:
                losses[tok] = 0.0
                continue

            temp_vocab = vocab_set - {tok}
            temp_probs = {t: probs[t] for t in temp_vocab}
            s = sum(temp_probs.values())
            for t in temp_probs:
                temp_probs[t] /= s

            ll_old = sum(word_likelihood(words[idx], probs, vocab_set) for idx in affected)
            ll_new = sum(word_likelihood(words[idx], temp_probs, temp_vocab) for idx in affected)

            losses[tok] = ll_old - ll_new

        remove_count = max(1, len(vocab) // 5)  # prune bottom 20%
        removable = sorted(losses.items(), key=lambda x: x[1])[:remove_count]

        for tok, _ in removable:
            vocab_set.remove(tok)
            del probs[tok]

        s = sum(probs.values())
        for t in probs:
            probs[t] /= s

        vocab = list(vocab_set)

    return vocab, probs

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_unigram_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def tokenize(text, tokenizer):
    vocab_set = set(tokenizer.keys())
    tokens = []
    i = 0
    while i < len(text):
        match = None
        for j in range(min(10, len(text)-i), 0, -1):
            sub = text[i:i+j]
            if sub in vocab_set:
                match = sub
                break
        if match is None:
            match = text[i]  
        tokens.append(match)
        i += len(match)
    return tokens


def detokenize(tokens, tokenizer):
    return "".join(tokens)

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

import time
if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "220688"

    train_text = load_training_data(args.train)
    vocab, tokenizer = train_unigram_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)

    total_time = time.time() - start_time
    print(f"Training complete. Time: {total_time:.2f}s")
