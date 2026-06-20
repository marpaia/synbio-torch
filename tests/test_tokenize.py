from __future__ import annotations

from synbiotorch.tokenize.char import CharTokenizer
from synbiotorch.tokenize.kmer import KmerTokenizer


def test_kmer_vocab_size():
    tok = KmerTokenizer(k=3)
    # 5 special tokens + 4^3 = 64 codons.
    assert tok.vocab_size == 5 + 64


def test_kmer_encode_wraps_with_special_tokens():
    tok = KmerTokenizer(k=2, max_length=64)
    enc = tok.encode("ACGT")
    cls, sep = tok._vocab["<cls>"], tok._vocab["<sep>"]
    assert enc.input_ids[0] == cls
    assert enc.input_ids[-1] == sep
    assert all(m == 1 for m in enc.attention_mask)
    assert len(enc.input_ids) == len(enc.attention_mask)


def test_kmer_unknown_base_maps_to_unk():
    tok = KmerTokenizer(k=2)
    enc = tok.encode("ANNA")  # contains ambiguous N
    assert tok._unk in enc.input_ids


def test_kmer_respects_max_length():
    tok = KmerTokenizer(k=1, max_length=8)
    enc = tok.encode("A" * 100)
    assert len(enc.input_ids) <= 8


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer(max_length=32)
    enc = tok.encode("ACGTN")
    # cls + 5 bases + sep
    assert len(enc.input_ids) == 7
    assert enc.input_ids[0] == tok._vocab["<cls>"]


def test_special_token_ids_cover_reserved_tokens():
    for tok in (KmerTokenizer(k=2), CharTokenizer()):
        assert tok.pad_token_id in tok.special_token_ids
        assert tok.mask_token_id in tok.special_token_ids
        assert len(tok.special_token_ids) == 5


def test_char_protein_alphabet():
    tok = CharTokenizer(alphabet="protein")
    # 5 special tokens + 25 amino-acid/ambiguity codes.
    assert tok.vocab_size == 5 + 25
    enc = tok.encode("MKWVTFISLLFLFSSAYS")  # all standard residues
    assert tok._unk not in enc.input_ids
    # A DNA-only char tokenizer would have a smaller, different vocabulary.
    assert tok.vocab_size != CharTokenizer(alphabet="dna").vocab_size
