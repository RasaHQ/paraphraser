import torch
from fairseq.models import FairseqEncoder, FairseqDecoder, FairseqEncoderDecoderModel
from fairseq import utils
from src.nmt_paraphraser.utils import (
    make_vocab_start_map,
    make_word_penalties_tokens,
)

"""
## Code adapted from https://github.com/thompsonb/prism/blob/master/paraphrase_generation/generate_paraphrases.py
"""


class NgramDownweightEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.mapx = make_vocab_start_map(self.dictionary.symbols)
        self.vocab_set = set(self.dictionary.symbols)
        self.args = args

    def forward(self, src_tokens, src_lengths):
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch. We discuss Tasks in the next tutorial, but for now just
        # know that *src_tokens* has shape `(batch, src_len)` and *src_lengths*
        # has shape `(batch)`.

        # Note that the source is typically padded on the left. This can be
        # configured by adding the `--left-pad-source "False"` command-line
        # argument, but here we'll make the Encoder handle either kind of
        # padding by converting everything to be right-padded.

        if self.args.left_pad_source:
            # Convert left-padding to right-padding.
            src_tokens = utils.convert_padding_direction(
                src_tokens, padding_idx=self.dictionary.pad(), left_to_right=True
            )

        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        debug_out = self.dictionary.string(
            src_tokens, bpe_symbol=None, escape_unk=False
        )

        batch_penalties = []
        for line in debug_out.split("\n"):
            penalties = make_word_penalties_tokens(
                line=line,
                vocab=self.vocab_set,
                mapx=self.mapx,
                dictionary=self.dictionary,
            )
            batch_penalties.append(penalties)

        return batch_penalties

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        # these maps arent going to be modified, so multiple references is fine
        return [encoder_out[ii] for ii in new_order.cpu().numpy()]


class NgramDownweightDecoder(FairseqDecoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.
    def forward(self, prev_output_tokens, encoder_out):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """

        batch_size = prev_output_tokens.shape[0]
        tgt_len = prev_output_tokens.shape[1]
        vocab_size = len(self.dictionary)

        xx = torch.zeros([batch_size, tgt_len, vocab_size], dtype=torch.float32)

        penalties = encoder_out[0]

        prefix = ()

        indices = penalties[prefix][0]

        xx[:, -1, indices] -= self.args.prism_a * 1 ** self.args.prism_b

        # Return the logits and ``None`` for the attention weights
        xx = xx.cuda() if not self.args.cpu else xx
        return xx, None


class NgramDownweightModel(FairseqEncoderDecoderModel):
    def max_positions(self):
        return (123456, 123456)

    @classmethod
    def build_model(cls, args, task):
        # Initialize our Encoder and Decoder.
        encoder = NgramDownweightEncoder(args=args, dictionary=task.source_dictionary)
        decoder = NgramDownweightDecoder(args=args, dictionary=task.target_dictionary)
        model = NgramDownweightModel(encoder, decoder)

        return model

    def get_normalized_probs(self, decoder_out, log_probs):
        return decoder_out[0]

    def reset_params(self, prism_a=0.1, prism_b=4):
        self.decoder.args.prism_a = prism_a
        self.decoder.args.prism_b = prism_b
