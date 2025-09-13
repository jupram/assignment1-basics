import os
from typing import BinaryIO
import regex as re
import multiprocessing as mp
from multiprocessing import cpu_count
import time


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(
    input_path: str, start: int, end: int, special_token: bytes) -> dict[bytes, int]:
    """
    Process the specified chunk of the input file.
    Splits the chunk by the decoded special token, tokenizes each part using regex,
    counts tokens per part, merges them, and prints the results.
    Returns the merged tokens count.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Compile regex pattern
    pattern = re.compile(PAT)

    # Read and process the specified chunk
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # Split chunk using the special token (decoded)
        parts = chunk.split(special_token.decode("utf-8"))
        # Array to hold token counts for each part
        tokens_count = [{} for _ in range(len(parts))]
        
        for i, part in enumerate(parts):
            # Tokenize each part using regex
            counter = {}
            for match in pattern.finditer(part):
                token = match.group(0)
                counter[token] = counter.get(token, 0) + 1
            tokens_count[i] = counter
        
        # Merge all token count dictionaries into one
        merged_tokens_count = merge_token_counts(tokens_count)
        print(f"Processed chunk from {start} to {end} in process {mp.current_process().name}")
        return merged_tokens_count

def merge_token_counts(counts_list: list[dict[bytes, int]]) -> dict[bytes, int]:
    """
    Merge a list of token count dictionaries into a single dictionary.
    """
    merged_counts = {}
    for counts in counts_list:
        for token, count in counts.items():
            merged_counts[token] = merged_counts.get(token, 0) + count
    return merged_counts

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from scratch on the given input file.
    Returns the vocabulary and merges.
    """
     # check if input_path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    
    # first intialize empty dictionary for tokens_count for each chunk
    tokens_count = []
    # create a process thread for each chunk
    num_processes = kwargs.get("num_processes", cpu_count())
    special_tokens = [token.encode("utf-8") for token in special_tokens]
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0])
    print(f"Chunk boundaries: {chunk_boundaries}")

    # Prepare arguments for each process
    args = [
        (input_path, start, end, special_tokens[0])
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    ]

    with mp.Pool(processes=num_processes) as pool:
        tokens_count = pool.starmap(process_chunk, args)

    final_tokens_count = merge_token_counts(tokens_count)
    print(f"Merged token counts from all chunks. Total unique tokens: {len(final_tokens_count)}")
    del tokens_count  # free memory
    
    # change the keys from bytes to tuple of bytes
    final_tokens_tuple_count = {tuple(bytes([b]) for b in token.encode("utf-8")): count for token, count in final_tokens_count.items()}
    del final_tokens_count
    # #print count for "the"
    # the = tuple(bytes([b]) for b in b" the")
    # print(f" Count for token {the}: {final_tokens_tuple_count.get(the, 0)}")

    # iterate over the items of final_tokens_tuple_count and read every tuple([bytes]) and for every consecutive pair of bytes, 
    # add to a new dictionary with count as value 
    pair_counts = {}
    for token_tuple, count in final_tokens_tuple_count.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    
    #print(pair_counts[(b' ', b't')], pair_counts[(b't', b'h')], pair_counts[(b'h', b'e')])

    # sort the pair_counts by value in descending order and key in descending order
    if pair_counts:
        most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        most_frequent_pair = most_frequent_pair[0]
    else:
        most_frequent_pair = None

    # Initialize vocabulary with specail tokens and single-character tokens
    # add sepcial tokens to vocab
    vocab = {}
    vocab_size_current = 0
    for token in special_tokens:
        vocab[vocab_size_current] = token
        vocab_size_current += 1
    # add single character tokens from ascii range 0 to 255
    for i in range(256):
        vocab[vocab_size_current] = bytes([i])
        vocab_size_current += 1
    


    merges = []
    # Perform BPE merges until reaching the desired vocabulary size
    while vocab_size_current < vocab_size and most_frequent_pair: 
        # Get the most frequent pair
        merges.append(most_frequent_pair)

        # Create new token by merging the most frequent pair
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        # Add new token to vocabulary
        print(f"Adding new token to vocab: {new_token} with size {len(new_token)}")
        vocab[vocab_size_current] = new_token
    
        vocab_size_current += 1

        # Update token counts with the new merged token
        keys_changed = {}
        
        for token_tuple, count in final_tokens_tuple_count.items():
            # Replace occurrences of the most frequent pair with the new token
            new_token_tuple = []
            i = 0
            merged = False
            while i < len(token_tuple):
                if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == most_frequent_pair:
                    new_token_tuple.append(new_token)
                    i += 2
                    merged = True
                else:
                    new_token_tuple.append(token_tuple[i])
                    i += 1
            if merged:
                keys_changed[token_tuple] = new_token_tuple
                
        # create a new dictionary with the items from final_tokens_tuple_count that were changed
        for key in keys_changed.items():
            count = final_tokens_tuple_count.pop(key[0])
            final_tokens_tuple_count[tuple(key[1])] = final_tokens_tuple_count.get(tuple(key[1]), 0) + count
        # Recalculate pair counts
        pair_counts = {}
        for token_tuple, count in final_tokens_tuple_count.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count    

        # Sort pairs by frequency and lexicographically
        if pair_counts:
            most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
            most_frequent_pair = most_frequent_pair[0]
        else:
            most_frequent_pair = None
        #print(f"Vocab size: {vocab_size_current}, Most common pair: {sorted_pairs[0] if sorted_pairs else 'N/A'}") 

    return vocab, merges
 
def main():
    input_path = r"data/TinyStoriesV2-GPT4-train.txt"
    start = time.time()
    v, m =run_train_bpe(
        input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    end = time.time()

    print(f"Elapsed time: {end - start:.2f} seconds")
    print(f"Final vocab size: {len(v)}")

    #print the longest token in the vocab
    longest_token = max(v.values(), key=len)
    print(f"Longest token in vocab: {longest_token} with length {len(longest_token)}")

    #serialize the vocab and merges to json and txt files respectively
    import json
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v.decode("utf-8", errors="ignore") for k, v in v.items()}, f, ensure_ascii=False, indent=4)
    with open("merges.txt", "w", encoding="utf-8") as f:
        for merge in m:
            f.write(f"{merge[0].decode('utf-8', errors='ignore')} {merge[1].decode('utf-8', errors='ignore')}\n")

if __name__ == "__main__":
    main()
    