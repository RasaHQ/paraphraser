import os

import torch
from fairseq import bleu, checkpoint_utils, progress_bar, tasks
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq import utils
from fairseq.sequence_generator import EnsembleModel
import sentencepiece as spm
import time

from paraphraser.utils import run_bash_cmd, Bunch
from paraphraser.modelling.utils import forward_decoder, get_final_string
from paraphraser.modelling.ngram_downweight_model_starting import NgramDownweightModel


class NMTParaphraser:

    def __init__(self, args, lite_mode=True):

        if lite_mode:
            EnsembleModel.forward_decoder = forward_decoder
        self.args = Bunch(args)
        self._load_tokenizer()
        self._load_model(lite_mode)

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
            self.src_dict = getattr(self.task, 'source_dictionary', None)
        except NotImplementedError:
            self.src_dict = None
        self.tgt_dict = self.task.target_dictionary

        # Load ensemble
        print('| loading model(s) from {}'.format(self.args.path))
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(':'),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )

        # Optimize ensemble for generation
        for model in self.models:

            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        if not lite_mode:
            self.ngram_downweight_model = NgramDownweightModel.build_model(self.args, self.task)
            self.models.append(self.ngram_downweight_model)  # ensemble Prism multilingual NMT model and model to downweight n-grams

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
        self.sp.Load(os.path.join(self.args.model_dir, 'spm.model'))

    def tokenize(self, sentences, lang):

        sp_sents = [' '.join(self.sp.EncodeAsPieces(sent)) for sent in sentences]
        with open('test.src', 'wt') as fout:
            for sent in sp_sents:
                fout.write(sent + '\n')

        # we also need a dummy output file with the language tag
        with open('test.tgt', 'wt') as fout:
            for sent in sp_sents:
                fout.write(f'<{lang}> \n')

    def preprocess_nmt(self):

        run_bash_cmd("rm -rf data-bin")
        run_bash_cmd("rm -rf test_bin")
        run_bash_cmd(
            f"fairseq-preprocess --source-lang src --target-lang tgt --joined-dictionary --srcdict {self.args.model_dir}/dict.tgt.txt --trainpref test --validpref test --testpref test --destdir test_bin")

    def pass_decoder(self):

        self.task.load_dataset(self.args.gen_subset)
        itr = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.gen_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(), *self.fixed_max_positions),
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
                if 'net_input' not in sample:
                    continue

                prefix_tokens = None
                if self.args.prefix_size > 0:
                    prefix_tokens = sample['target'][:, :self.args.prefix_size]

                hypos = self.task.inference_step(generator, self.models, sample, prefix_tokens)

                for i, sample_id in enumerate(sample['id'].tolist()):

                    # Remove padding
                    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], self.tgt_dict.pad())
                    src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)

                    # Process top predictions
                    for j, hypo in enumerate(hypos[i][:self.args.nbest]):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'],
                            align_dict=self.align_dict,
                            tgt_dict=self.tgt_dict,
                            remove_bpe=self.args.remove_bpe,
                        )
                        well_formed_str = get_final_string(hypo_str)
                        all_preds.append((sample_id, well_formed_str))

        return all_preds

    def generate_paraphrase(self, paragraph, lang, prism_a=0.01, prism_b=4):

        self.reset_prism_value(prism_a, prism_b)
        start = time.time()
        self.tokenize([paragraph], lang)
        self.preprocess_nmt()
        paraphrases = self.pass_decoder()
        paraphrases = sorted(paraphrases, key=lambda x: x[0])
        # remove the id here
        paraphrases = [p[1].replace(f"<{lang}>", "").strip() for p in paraphrases]
        paraphrases = [p.replace(f"<unk>", "").strip() for p in paraphrases]
        end = time.time()
        print("Complete inference time", end - start)
        return paraphrases

    def reset_prism_value(self, prism_a, prism_b):

        if len(self.models) > 1:
            self.models[-1].reset_params(prism_a, prism_b)
