"""
Standalone cleaning script for FastMTP self-distilled data.

Operates on the final generated JSONL file. Applies:
  1. MinHash cross-source deduplication (Jaccard ~0.85)
  2. Incomplete/truncated reasoning chain filtering
  3. Excessive repetition filtering
  4. Length range filtering

Usage:
  python scripts/clean_data.py --input data/distilled_data.jsonl --output data/train_data_cleaned.jsonl
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


# ============================================================================
# Config
# ============================================================================

# MinHash parameters
MINHASH_PERMUTATIONS = 128   # hash functions
MINHASH_JACCARD_THRESHOLD = 0.85  # remove if similarity >= 85%

# Repetition threshold (max fraction of identical 5-grams)
REPETITION_MAX_NGRAM_RATIO = 0.3

# Length bounds (characters)
PROMPT_MIN_LEN = 10
PROMPT_MAX_LEN = 16000
RESPONSE_MIN_LEN = 10
RESPONSE_MAX_LEN = 24000

# Output
DEFAULT_INPUT = "/ssd/yangxw/FastMTP/data/distilled_data.jsonl"
DEFAULT_OUTPUT = "/ssd/yangxw/FastMTP/data/train_data_cleaned.jsonl"


# ============================================================================
# 1. MinHash cross-source dedup
# ============================================================================

def tokenize_for_minhash(text, n=5):
    """Tokenize text into character n-grams for MinHash."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def minhash_deduplicate(samples, jaccard_threshold=MINHASH_JACCARD_THRESHOLD):
    """
    Cross-source MinHash deduplication.
    Groups near-duplicate samples by Jaccard similarity and keeps one per cluster.
    """
    print("\n[1/5] MinHash deduplication...")
    print(f"  Permutations: {MINHASH_PERMUTATIONS}, "
          f"Jaccard threshold: {jaccard_threshold}")

    if len(samples) < 2:
        print("  Too few samples, skipping.")
        return samples

    lsh = MinHashLSH(threshold=jaccard_threshold,
                     num_perm=MINHASH_PERMUTATIONS)
    minhashes = []
    # (idx, representative_idx) for union-find style clustering
    parent = list(range(len(samples)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build MinHash signatures + query LSH
    print("  Computing MinHash signatures...")
    for i, sample in enumerate(tqdm(samples, desc="  MinHash sign")):
        text = sample["messages"][0]["content"] + " " + sample["messages"][1]["content"]
        tokens = tokenize_for_minhash(text)
        m = MinHash(num_perm=MINHASH_PERMUTATIONS)

        if len(tokens) < 5:
            # Very short text — use word-level hash to produce valid signature
            for w in text.split():
                m.update(w.encode("utf8"))
                if m.is_empty() is False:
                    break
            # If still empty, use a dummy token
            if m.is_empty():
                m.update(b" ")
        else:
            for t in tokens:
                m.update(t.encode("utf8"))

        minhashes.append(m)

        # Query LSH for near-duplicates
        candidates = lsh.query(m)
        for j in candidates:
            if j < i:
                estimated_jaccard = m.jaccard(minhashes[j])
                if estimated_jaccard >= jaccard_threshold:
                    union(i, j)

        lsh.insert(i, m)

    # Build cluster representatives
    clusters = defaultdict(list)
    for i in range(len(samples)):
        clusters[find(i)].append(i)

    deduped = []
    removed = 0
    for root, members in clusters.items():
        deduped.append(samples[members[0]])  # keep first
        removed += len(members) - 1

    print(f"  Clusters: {len(clusters)}, removed: {removed} duplicates, "
          f"kept: {len(deduped)}")
    return deduped


# ============================================================================
# 2. Incomplete / truncated response filtering
# ============================================================================

def has_complete_prose(response):
    """General: ends with sentence-ending punctuation."""
    return bool(re.search(r'[.!?\"\'\u3002\uff01\uff1f\u201d\u2019]\s*$', response.strip()))


def has_complete_math(response):
    """Math: contains boxed answer or final statement."""
    patterns = [
        r'\\boxed\{',           # LaTeX boxed
        r'(?i)answer\s*[:=]\s*',  # "answer: ..."
        r'(?i)therefore\b.*[.!]',  # concluding statement
        r'(?i)the answer is\b',
        r'(?i)final answer\b',
    ]
    return any(re.search(p, response) for p in patterns)


def has_complete_code(response):
    """Code: balanced braces and proper closure."""
    # Check brace balance
    braces = {'{': '}', '[': ']', '(': ')'}
    stack = []
    for ch in response:
        if ch in braces:
            stack.append(braces[ch])
        elif ch in braces.values():
            if stack and stack[-1] == ch:
                stack.pop()
    braces_balanced = len(stack) == 0

    # Check for proper ending (triple-backtick, or code ends cleanly)
    ends_cleanly = bool(re.search(
        r'(```\s*$|return\b.*\n|end\b.*\n|\}\s*$|\)\s*$|\]\s*$)', response
    ))

    return braces_balanced or ends_cleanly


def has_llm_refusal(response):
    """Model refusal patterns."""
    refusal_pats = [
        r'(?i)\b(sorry|i apologize|i cannot|i\'m unable)\b',
        r'(?i)\b(as an ai|as a language model)\b',
        r'(?i)(cannot provide|not able to|unable to (fulfill|complete|answer))',
    ]
    return any(re.search(p, response) for p in refusal_pats)


def is_truncated(response):
    """Detect mid-sentence truncation."""
    # Obvious truncation markers
    if re.search(r'(?i)(未完待续|to be continued|continued\.{3}|\.{3}\s*$)', response):
        return True
    # Ends mid-word (e.g., "the quick brown fo")
    if re.search(r'\s\w{1,4}$', response) and not re.search(r'[.!?\"\'\]\)\u3002\uff01\uff1f]$', response):
        return True
    return False


def filter_incomplete(samples):
    """Filter samples with incomplete, truncated, or refusal responses."""
    print("\n[2/5] Filtering incomplete responses...")

    kept = []
    stats = defaultdict(int)
    total = len(samples)

    for s in tqdm(samples, desc="  Checking"):
        resp = s["messages"][1]["content"].strip()
        prompt = s["messages"][0]["content"].strip()

        # Too short
        if len(resp) < RESPONSE_MIN_LEN:
            stats["too_short"] += 1
            continue

        # LLM refusal
        if has_llm_refusal(resp[-500:]):  # check last 500 chars
            stats["refusal"] += 1
            continue

        # Truncation
        if is_truncated(resp):
            stats["truncated"] += 1
            continue

        # Domain-specific completeness (best-effort)
        # Detect if prompt asks for code (contains code-like markers)
        prompt_asks_code = bool(re.search(
            r'(?i)(write.*(code|function|program|script|class|def )|implement|'
            r'```|import\s+\w+|def\s+\w+\()', prompt
        ))
        prompt_asks_math = bool(re.search(
            r'(?i)(solve|calculate|compute|prove|derive|find the|evaluate|'
            r'what is the (value|sum|product|result)|\\?frac|\\?sqrt)',
            prompt
        ))

        if prompt_asks_code and not has_complete_code(resp):
            stats["incomplete_code"] += 1
            continue

        if prompt_asks_math and not has_complete_math(resp):
            stats["incomplete_math"] += 1
            continue

        # General prose: only flag very obviously incomplete (mid-word)
        if not prompt_asks_code and not prompt_asks_math:
            if is_truncated(resp) and len(resp) < 200:
                stats["incomplete_prose"] += 1
                continue

        kept.append(s)

    print(f"  Total: {total}")
    for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count} removed")
    print(f"  Kept: {len(kept)}")
    return kept


# ============================================================================
# 3. Excessive repetition
# ============================================================================

def filter_repetitive(samples):
    """Filter samples with excessive repeated n-grams."""
    print("\n[3/5] Filtering repetitive content...")

    kept = []
    removed = 0
    for s in tqdm(samples, desc="  Checking"):
        words = s["messages"][1]["content"].split()
        if len(words) < 20:
            kept.append(s)
            continue

        # 5-gram repetition check
        ngrams = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
        if ngrams:
            most_common_count = Counter(ngrams).most_common(1)[0][1]
            ratio = most_common_count / len(ngrams)
            if ratio > REPETITION_MAX_NGRAM_RATIO:
                removed += 1
                continue

        # Also check character-level: long runs of identical chars
        text = s["messages"][1]["content"]
        longest_run = 0
        current_run = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                current_run += 1
            else:
                longest_run = max(longest_run, current_run)
                current_run = 1
        longest_run = max(longest_run, current_run)
        if longest_run > 200:  # >200 repeated chars
            removed += 1
            continue

        kept.append(s)

    print(f"  Removed: {removed}, kept: {len(kept)}")
    return kept


# ============================================================================
# 4. Length filtering
# ============================================================================

def filter_length(samples):
    """Filter by prompt and response length."""
    print("\n[4/5] Filtering by length...")

    kept = []
    removed_short = 0
    removed_long = 0
    for s in tqdm(samples, desc="  Checking"):
        pl = len(s["messages"][0]["content"])
        rl = len(s["messages"][1]["content"])
        if pl < PROMPT_MIN_LEN or rl < RESPONSE_MIN_LEN:
            removed_short += 1
            continue
        if pl > PROMPT_MAX_LEN or rl > RESPONSE_MAX_LEN:
            removed_long += 1
            continue
        kept.append(s)

    print(f"  Too short: {removed_short}, too long: {removed_long}, kept: {len(kept)}")
    return kept


# ============================================================================
# 5. Exact dedup (final pass after MinHash)
# ============================================================================

def exact_dedup(samples):
    """Final exact dedup pass to catch any remaining identical samples."""
    print("\n[5/5] Exact dedup (final pass)...")
    seen = set()
    kept = []
    removed = 0
    for s in tqdm(samples, desc="  Checking"):
        h = hashlib.md5(
            (s["messages"][0]["content"] + s["messages"][1]["content"]).encode()
        ).hexdigest()
        if h not in seen:
            seen.add(h)
            kept.append(s)
        else:
            removed += 1
    print(f"  Removed: {removed}, kept: {len(kept)}")
    return kept


# ============================================================================
# Main
# ============================================================================

def clean(input_file, output_file, jaccard_threshold=MINHASH_JACCARD_THRESHOLD):
    print("=" * 60)
    print("FastMTP Data Cleaning")
    print(f"Jaccard threshold: {jaccard_threshold}")
    print("=" * 60)

    input_file = Path(input_file)
    if not input_file.exists():
        print(f"ERROR: {input_file} not found.")
        sys.exit(1)

    # Load
    with open(input_file) as f:
        samples = [json.loads(line) for line in f]
    print(f"\nLoaded: {len(samples)} samples from {input_file}")

    pl = [len(s["messages"][0]["content"]) for s in samples]
    rl = [len(s["messages"][1]["content"]) for s in samples]
    print(f"  Prompt:   min={min(pl)}, max={max(pl)}, avg={sum(pl)/len(pl):.0f} chars")
    print(f"  Response: min={min(rl)}, max={max(rl)}, avg={sum(rl)/len(rl):.0f} chars")

    # Pipeline
    samples = minhash_deduplicate(samples, jaccard_threshold=jaccard_threshold)
    samples = filter_incomplete(samples)
    samples = filter_repetitive(samples)
    samples = filter_length(samples)
    samples = exact_dedup(samples)

    # Save
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    pl = [len(s["messages"][0]["content"]) for s in samples]
    rl = [len(s["messages"][1]["content"]) for s in samples]

    print(f"\n{'='*60}")
    print(f"Cleaned: {len(samples)} samples → {output_file}")
    print(f"  Avg prompt:   {sum(pl)/len(pl):.0f} chars")
    print(f"  Avg response: {sum(rl)/len(rl):.0f} chars")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="FastMTP Data Cleaner")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL file")
    parser.add_argument("--jaccard", type=float, default=MINHASH_JACCARD_THRESHOLD,
                        help="MinHash Jaccard threshold (0-1)")
    args = parser.parse_args()

    jaccard_threshold = args.jaccard
    clean(args.input, args.output, jaccard_threshold=jaccard_threshold)


if __name__ == "__main__":
    main()
