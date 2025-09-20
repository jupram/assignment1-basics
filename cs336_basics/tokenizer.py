import os
from typing import BinaryIO
import regex as re
import multiprocessing as mp
from multiprocessing import cpu_count
import time
from typing import IO, Any, BinaryIO, Iterable, Iterator



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

class Tokenizer:
    """A BPE tokenizer that uses a provided vocabulary, merges, and special tokens."""
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.inv_vocab = {t: id for id, t in self.vocab.items()}
        #assign index to UNK token if not present
        self.unk_id = max(self.vocab.keys()) + 1
        vocab[self.unk_id] = b"<|unk|>"
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._pretokenize_regex = re.compile(PAT)
    
    @classmethod
    def from_files(cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Create a Tokenizer instance from vocab and merges files."""
        # Load vocab
        vocab = {}
        #parse json file with int keys and string values
        import json
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
            for token_id, token in vocab_json.items():
                vocab[int(token_id)] = token.encode("utf-8")
        
        # Load merges
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            #parse text json array of arrays with two string elements each
            merge_json = json.load(f)
            for m in merge_json:
                token1, token2 = m
                merges.append((token1.encode("utf-8"), token2.encode("utf-8")))
        
        return cls(vocab, merges, special_tokens)

    def _do_single_merge(self, tokens: list[bytes]) -> list[bytes]:
        """Perform a single BPE merge on the list of tokens."""
        # Create a dictionary of pairs of adjacent tokens
        pairs = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

        # Find the first mergeable pair according to the merges list
        for merge in self.merges:
            if merge in pairs:
                # merge all occurrences of the pair
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2  # Skip the next token as it's merged
                    else:
                        new_tokens.append(tokens[i])
                        i += 1 # Move to the next token
                return new_tokens

        # If no merges were performed, return the original tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a list of token IDs."""
        # Pre-tokenize the input text
        tokens = self._pretokenize(text)
        token_ids = []
        #print(tokens)
        for t in tokens:
            if t in [k.encode("utf-8") for k in self.special_tokens]:
                token_id = self.inv_vocab.get(t, self.unk_id)
                if token_id is not None:
                    token_ids.append(token_id)
                continue
            else:
                # turn token into list of bytes
                subtokens = [bytes([b]) for b in t]
                # Repeatedly apply merges until no more merges can be applied
                while True:
                    new_subtokens = self._do_single_merge(subtokens)
                    if new_subtokens == subtokens:
                        # Convert subtokens to token IDs and add to token_ids
                        for subtoken in new_subtokens:
                            token_id = self.inv_vocab.get(subtoken, self.unk_id)
                            if token_id is not None:
                                token_ids.append(token_id)
                        break
                    subtokens = new_subtokens
        #print(token_ids)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings into an iterator of token IDs."""
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        print(token_ids)
        tokens = [self.vocab.get(tid) for tid in token_ids]
        # if any token is b<|unk|>, replace it with U+FFFD
        for i, token in enumerate(tokens):
            if token == b"<|unk|>":
                tokens[i] = b"\xef\xbf\xbd"  # U+FFFD in UTF-8
        #turn list of bytes into list of strings
        tokens = b"".join(tokens)
        return tokens.decode("utf-8", errors='replace')

    def _pretokenize(self, text: str) -> list[bytes]:
        """Pre-tokenize the input text into a list of byte tokens."""
        # create a regex pattern that matches either any of the special tokens or the pretokenize regex
              
        #special_tokens = "|".join(map(re.escape, self.special_tokens))
        special_tokens = "|".join(sorted(map(re.escape, self.special_tokens), key=len, reverse=True))
        tokens = []
        parts = re.split(f"({special_tokens})", text)
        print(parts)
        for part in parts:
            if part in self.special_tokens:
                tokens.append(part.encode("utf-8"))
            else:
            # Tokenize each part using regex
                for match in self._pretokenize_regex.finditer(part):
                    token = match.group(0)
                    tokens.append(token.encode("utf-8"))
        return tokens


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError
    
def main():
    tokenizer_vocab_path = "vocab_s.json"
    tokenizer_merges_path = "merges_s.json"
    special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]

    tokenizer = Tokenizer.from_files(
        vocab_path=tokenizer_vocab_path,
        merges_path=tokenizer_merges_path,
        special_tokens=special_tokens
    )
    #german text with diactritics and emojis
    text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)
    print(text == decoded_text )

if __name__ == "__main__":
    main()
    