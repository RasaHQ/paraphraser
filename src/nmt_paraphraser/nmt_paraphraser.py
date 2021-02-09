import os
import logging

import torch
from fairseq import checkpoint_utils, progress_bar, tasks
from fairseq import utils
from fairseq.sequence_generator import EnsembleModel
import sentencepiece as spm
from collections import defaultdict, OrderedDict

from src.utils import run_bash_cmd, Bunch, merge_dicts
from src.nmt_paraphraser.utils import forward_decoder, get_final_string
from src.nmt_paraphraser.ngram_downweight_model import NgramDownweightModel

"""
## Code adapted from https://github.com/thompsonb/prism/blob/master/paraphrase_generation/generate_paraphrases.py
"""

logger = logging.getLogger()

model_dir = "m39v1/"
default_args = {
    "no_progress_bar": False,
    "log_interval": 1000,
    "log_format": "tqdm",
    "tensorboard_logdir": "",
    "seed": 1,
    "fp16_init_scale": 128,
    "fp16_scale_window": None,
    "fp16_scale_tolerance": 0.0,
    "min_loss_scale": 0.0001,
    "threshold_loss_scale": None,
    "user_dir": None,
    "empty_cache_freq": 0,
    "criterion": "cross_entropy",
    "tokenizer": None,
    "bpe": None,
    "optimizer": "nag",
    "lr_scheduler": "fixed",
    "task": "translation",
    "num_workers": 1,
    "skip_invalid_size_inputs_valid_test": True,
    "max_tokens": None,
    "max_sentences": 8,
    "required_batch_size_multiple": 8,
    "dataset_impl": None,
    "gen_subset": "test",
    "num_shards": 1,
    "shard_id": 0,
    "path": os.path.join(model_dir, "checkpoint.pt"),
    "remove_bpe": None,
    "quiet": False,
    "model_overrides": "{}",
    "results_path": None,
    "max_len_a": 0,
    "max_len_b": 40,
    "min_len": 1,
    "match_source_len": False,
    "no_early_stop": False,
    "unnormalized": False,
    "no_beamable_mm": False,
    "lenpen": 1,
    "unkpen": 0,
    "replace_unk": None,
    "sacrebleu": False,
    "score_reference": False,
    "prefix_size": 1,
    "no_repeat_ngram_size": 0,
    "sampling": False,
    "sampling_topk": -1,
    "sampling_topp": -1.0,
    "print_alignment": False,
    "print_step": False,
    "iter_decode_eos_penalty": 0.0,
    "iter_decode_max_iter": 10,
    "iter_decode_force_max_iter": False,
    "retain_iter_history": False,
    "decoding_format": None,
    "momentum": 0.99,
    "weight_decay": 0.0,
    "force_anneal": None,
    "lr_shrink": 0.1,
    "warmup_updates": 0,
    "data": "test_bin",
    "source_lang": None,
    "target_lang": None,
    "lazy_load": False,
    "raw_text": False,
    "load_alignments": False,
    "left_pad_source": False,
    "left_pad_target": "False",
    "upsample_primary": 1,
    "truncate_source": False,
    "model_dir": model_dir,
}


class NMTParaphraser:
    def __init__(self, run_args, lite_mode=True):

        if lite_mode:
            EnsembleModel.forward_decoder = forward_decoder

        run_args = merge_dicts(default_args, vars(run_args))
        self._fill_hardware_args(run_args)
        self.args = Bunch(run_args)
        self._load_tokenizer()
        self._load_model(lite_mode)

    @staticmethod
    def _fill_hardware_args(args):
        gpu = torch.cuda.is_available()
        args["cpu"] = not gpu
        args["fp16"] = True if gpu else False
        args["memory_efficient_fp16"] = False

    def _load_model(self, lite_mode):

        utils.import_user_module(self.args)

        if self.args.max_tokens is None and self.args.max_sentences is None:
            self.args.max_tokens = 12000

        self.use_cuda = torch.cuda.is_available() and not self.args.cpu

        # Load dataset splits
        self.task = tasks.setup_task(self.args)
        self.task.load_dataset(self.args.gen_subset)

        # Set dictionaries
        try:
            self.src_dict = getattr(self.task, "source_dictionary", None)
        except NotImplementedError:
            self.src_dict = None
        self.tgt_dict = self.task.target_dictionary

        # Load ensemble
        logger.info(f"| loading model(s) from {self.args.path}")
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(":"),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )

        # Optimize ensemble for generation
        for model in self.models:

            model.make_generation_fast_(
                beamable_mm_beam_size=None
                if self.args.no_beamable_mm
                else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        if not lite_mode:
            self.ngram_downweight_model = NgramDownweightModel.build_model(
                self.args, self.task
            )
            self.models.append(
                self.ngram_downweight_model
            )  # ensemble Prism multilingual NMT model and model to downweight n-grams

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        max_positions = [model.max_positions() for model in self.models]
        self.fixed_max_positions = []
        for x in max_positions:
            try:
                self.fixed_max_positions.append((x[0], x[1]))
            except:
                self.fixed_max_positions.append((12345677, x))

    def _load_tokenizer(self):

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(os.path.join(self.args.model_dir, "spm.model"))

    def tokenize(self, sentences, lang):

        sp_sents = [" ".join(self.sp.EncodeAsPieces(sent)) for sent in sentences]
        with open("test.src", "wt") as fout:
            for sent in sp_sents:
                fout.write(sent + "\n")

        # we also need a dummy output file with the language tag
        with open("test.tgt", "wt") as fout:
            for sent in sp_sents:
                fout.write(f"<{lang}> \n")

    def preprocess_nmt(self):

        run_bash_cmd("rm -rf data-bin")
        run_bash_cmd("rm -rf test_bin")
        run_bash_cmd(
            f"fairseq-preprocess --source-lang src --target-lang tgt --joined-dictionary --srcdict {self.args.model_dir}/dict.tgt.txt --trainpref test --validpref test --testpref test --destdir test_bin"
        )

    def pass_decoder(self):

        self.task.load_dataset(self.args.gen_subset)
        itr = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.gen_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(), *self.fixed_max_positions
            ),
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            num_shards=self.args.num_shards,
            shard_id=self.args.shard_id,
            num_workers=self.args.num_workers,
        ).next_epoch_itr(shuffle=False)

        # Initialize generator
        generator = self.task.build_generator(self.args)

        all_preds = []
        with progress_bar.build_progress_bar(self.args, itr) as t:
            for sample in t:
                sample = utils.move_to_cuda(sample) if self.use_cuda else sample
                if "net_input" not in sample:
                    continue

                prefix_tokens = None
                if self.args.prefix_size > 0:
                    prefix_tokens = sample["target"][:, : self.args.prefix_size]

                hypos = self.task.inference_step(
                    generator, self.models, sample, prefix_tokens
                )

                for i, sample_id in enumerate(sample["id"].tolist()):

                    # Remove padding
                    src_tokens = utils.strip_pad(
                        sample["net_input"]["src_tokens"][i, :], self.tgt_dict.pad()
                    )
                    src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)

                    # Process top predictions
                    for j, hypo in enumerate(hypos[i][: self.args.nbest]):
                        (
                            hypo_tokens,
                            hypo_str,
                            alignment,
                        ) = utils.post_process_prediction(
                            hypo_tokens=hypo["tokens"].int().cpu(),
                            src_str=src_str,
                            alignment=hypo["alignment"],
                            align_dict=self.align_dict,
                            tgt_dict=self.tgt_dict,
                            remove_bpe=self.args.remove_bpe,
                        )
                        well_formed_str = get_final_string(hypo_str)
                        all_preds.append((sample_id, well_formed_str))

        return all_preds

    def _post_process_paraphrases(
        self, all_paraphrases, lang_id,
    ):

        cleaned_paraphrases = []
        for paraphrases in all_paraphrases:
            paraphrases = [p.replace(f"<{lang_id}>", "").strip() for p in paraphrases]
            paraphrases = [p.replace(f"<unk>", "").strip() for p in paraphrases]
            cleaned_paraphrases.append(paraphrases)

        return cleaned_paraphrases

    def _fill_skipped_paraphrases(self, id_paraphrases, input_num_sentences):

        for index in range(input_num_sentences):
            if index not in id_paraphrases:
                id_paraphrases[index] = []
        return id_paraphrases

    def generate_paraphrase(self, sentences, lang, prism_a=0.01, prism_b=4):

        self.reset_prism_value(prism_a, prism_b)

        if isinstance(sentences, str):
            sentences = [sentences]

        self.tokenize(sentences, lang)
        self.preprocess_nmt()
        paraphrases = self.pass_decoder()
        paraphrases = sorted(paraphrases, key=lambda x: x[0])
        # Convert to a list of list where each internal list is a collection of paraphrases for one sentence.

        # Collect all paraphrases with the same sample id
        sample_id_paraphrases = defaultdict(list)

        for sample_id, paraphrase in paraphrases:
            sample_id_paraphrases[sample_id].append(paraphrase)

        sample_id_paraphrases = self._fill_skipped_paraphrases(sample_id_paraphrases, len(sentences))
        sample_id_paraphrases = OrderedDict(sorted(sample_id_paraphrases.items()))

        all_paraphrases = list(sample_id_paraphrases.values())
        all_paraphrases = self._post_process_paraphrases(all_paraphrases, lang)

        return all_paraphrases

    def reset_prism_value(self, prism_a, prism_b):

        if len(self.models) > 1:
            self.models[-1].reset_params(prism_a, prism_b)
