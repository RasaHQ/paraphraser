import torch
import math
from collections import defaultdict


@torch.no_grad()
def forward_decoder(self, tokens, encoder_outs, temperature=1.0):

    log_probs = []
    avg_attn = None
    for index, (model, encoder_out) in enumerate(zip(self.models, encoder_outs)):
        probs, attn = self._decode_one(
            tokens,
            model,
            encoder_out,
            self.incremental_states,
            log_probs=True,
            temperature=temperature,
        )
        log_probs.append(probs)
        if index == len(self.models) - 1:
            log_probs.append(torch.zeros_like(probs))
        if attn is not None:
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)

    avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(2)
    if avg_attn is not None:
        avg_attn.div_(len(self.models))
    return avg_probs, avg_attn


def get_final_string(hypo_str):
    str_tokens = hypo_str.split()
    final_string = ""
    for word in str_tokens:
        if word.startswith("▁"):
            word = word.replace("▁", "")
            final_string += f" {word}"
        else:
            final_string += word
    return final_string


def make_subword_penalties(line):
    """
    prefix: n-grams of subwords (n=1,2,3,4)
    penalize: the next subword
    """
    penalties = defaultdict(list)
    toks = line.replace("<pad>", "").split()
    for prefix_len in (0, 1, 2, 3):
        for ii in range(len(toks) - prefix_len):
            prefix = toks[ii : ii + prefix_len]
            next_word = toks[ii + prefix_len]
            penalties[tuple(prefix)].append((next_word, len(prefix)))

    return penalties


def make_word_penalties(line, vocab, mapx):
    """
    prefix: subwords that make up a n-gram (n=1,2,3,4) of FULL WORDS
    penalize: the next subword
    """
    from time import time

    t0 = time()

    penalties = defaultdict(list)
    uline = "▁"

    def breakup(tt):
        out = []
        for word in tt:
            for ii, subword in enumerate(word):
                if ii == 0:
                    out.append(uline + subword)
                else:
                    out.append(subword)
        return out

    line2 = line.replace("<pad>", "").strip()
    line2 = [
        x.replace("|", " ").strip().split()
        for x in line2.replace(" ", "|").split(uline)
        if x
    ]

    for prefix_len in (0, 1, 2, 3):
        for ii in range(len(line2) - prefix_len):
            prefix = line2[ii : ii + prefix_len]
            next_word = (
                uline + line2[ii + prefix_len][0]
            )  # just penalize starting a word, not continuing it

            whole_next_word = uline + "".join(line2[ii + prefix_len])

            word_prefix = breakup(prefix)

            # penalize any token that starts the next word, ignoring case
            # about 1s per line
            for tok in vocab:
                if whole_next_word.lower().startswith(tok.lower()):
                    penalties[tuple(word_prefix)].append((tok, len(prefix)))

            # build the longest part of the next word I can that is in the vocab
            longest_next_substring = uline
            for subthing in line2[ii + prefix_len]:
                if longest_next_substring + subthing in vocab:
                    longest_next_substring = longest_next_substring + subthing
                else:
                    break

            for tok in mapx[
                longest_next_substring
            ]:  # every word that starts the same, sans case
                penalties[tuple(word_prefix)].append((tok, len(prefix)))

    return penalties


def _add_to_penalty(penalties, word_prefix, tok, prefix):
    if tuple(word_prefix) not in penalties:
        penalties[tuple(word_prefix)] = {}

    if len(prefix) not in penalties[tuple(word_prefix)]:
        penalties[tuple(word_prefix)][len(prefix)] = []

    # if tok not in penalties[tuple(word_prefix)][len(prefix)]:
    penalties[tuple(word_prefix)][len(prefix)].append(tok)

    return penalties


def _add_to_penalty_token_ids(penalties, word_prefix, tok, prefix, dictionary):
    if tuple(word_prefix) not in penalties:
        penalties[tuple(word_prefix)] = {}

    if len(prefix) not in penalties[tuple(word_prefix)]:
        penalties[tuple(word_prefix)][len(prefix)] = []

    # if tok not in penalties[tuple(word_prefix)][len(prefix)]:
    penalties[tuple(word_prefix)][len(prefix)].append(dictionary.index(tok))

    return penalties


def make_word_penalties_tokens(line, vocab, mapx, dictionary):
    """
    prefix: subwords that make up a n-gram (n=1,2,3,4) of FULL WORDS
    penalize: the next subword
    """
    from time import time

    t0 = time()

    penalties = defaultdict(list)
    uline = "▁"

    def breakup(tt):
        out = []
        for word in tt:
            for ii, subword in enumerate(word):
                if ii == 0:
                    out.append(uline + subword)
                else:
                    out.append(subword)
        return out

    line2 = line.replace("<pad>", "").strip()
    line2 = [
        x.replace("|", " ").strip().split()
        for x in line2.replace(" ", "|").split(uline)
        if x
    ]

    for prefix_len in (0, 1, 2, 3):
        for ii in range(-1, len(line2) - prefix_len):
            prefix = line2[ii : ii + prefix_len + 1]
            next_word = (
                uline + line2[ii + prefix_len][0]
            )  # just penalize starting a word, not continuing it

            whole_next_word = uline + "".join(line2[ii + prefix_len])

            word_prefix = [dictionary.index(w) for w in breakup(prefix)]

            # penalize any token that starts the next word, ignoring case
            # about 1s per line
            for tok in vocab:
                if whole_next_word.lower().startswith(tok.lower()):
                    penalties = _add_to_penalty_token_ids(
                        penalties, word_prefix, tok, prefix, dictionary
                    )

            # build the longest part of the next word I can that is in the vocab
            longest_next_substring = uline
            for subthing in line2[ii + prefix_len]:
                if longest_next_substring + subthing in vocab:
                    longest_next_substring = longest_next_substring + subthing
                else:
                    break

            for tok in mapx[
                longest_next_substring
            ]:  # every word that starts the same, sans case
                penalties = _add_to_penalty_token_ids(
                    penalties, word_prefix, tok, prefix, dictionary
                )

    return penalties


def make_vocab_start_map(vocab):
    vocab_set = set(vocab)

    # build mapping from every lowercase subword to every cased variant in vocabulary
    ucase2case = defaultdict(set)
    for word in vocab:
        for ii in range(1, len(word) + 1):
            subword = word[:ii]
            if subword in vocab_set:
                ucase2case[subword.lower()].add(subword)

    # build mapping from every word to every prefix that starts that word (where "starts" ignores case)
    mapx = dict()
    for word in vocab:
        toks = set()
        for ii in range(1, len(word) + 1):
            subword = word[:ii]
            for fubar in ucase2case[subword.lower()]:
                toks.add(fubar)
        mapx[word] = list(toks)

    return mapx
