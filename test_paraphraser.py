#!/usr/bin/env python
# coding: utf-8

# In[1]:


from paraphraser.modelling.nmt_paraphraser import NMTParaphraser
import os
import time


# In[2]:


prism_a = 0.0
prism_b = 4
diverse_beam_groups = 10
beam_size = 10
beam_best = 10
diverse_beam_strength = 0.5
cpu = True
fp16 = False
mem_efficient_fp16 = False
left_pad_source = False
max_source_positions = 50
max_target_positions = 50
temperature = 1
match_source_len = False
replace_unk = None
model_dir = "m39v1/"
model_path = os.path.join(model_dir, "checkpoint.pt")
args = {
    "no_progress_bar": False,
    "log_interval": 1000,
    "log_format": None,
    "tensorboard_logdir": "",
    "seed": 1,
    "cpu": cpu,
    "fp16": fp16,
    "memory_efficient_fp16": mem_efficient_fp16,
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
    "skip_invalid_size_inputs_valid_test": False,
    "max_tokens": None,
    "max_sentences": 8,
    "required_batch_size_multiple": 8,
    "dataset_impl": None,
    "gen_subset": "test",
    "num_shards": 1,
    "shard_id": 0,
    "path": model_path,
    "remove_bpe": None,
    "quiet": False,
    "model_overrides": "{}",
    "results_path": None,
    "beam": beam_size,
    "nbest": beam_best,
    "max_len_a": 0,
    "max_len_b": 200,
    "min_len": 1,
    "match_source_len": match_source_len,
    "no_early_stop": False,
    "unnormalized": False,
    "no_beamable_mm": False,
    "lenpen": 1,
    "unkpen": 0,
    "replace_unk": replace_unk,
    "sacrebleu": False,
    "score_reference": False,
    "prefix_size": 1,
    "no_repeat_ngram_size": 0,
    "sampling": False,
    "sampling_topk": -1,
    "sampling_topp": -1.0,
    "temperature": temperature,
    "diverse_beam_groups": diverse_beam_groups,
    "diverse_beam_strength": diverse_beam_strength,
    "print_alignment": False,
    "print_step": False,
    "iter_decode_eos_penalty": 0.0,
    "iter_decode_max_iter": 10,
    "iter_decode_force_max_iter": False,
    "retain_iter_history": False,
    "decoding_format": None,
    "prism_a": prism_a,
    "prism_b": prism_b,
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
    "left_pad_source": left_pad_source,
    "left_pad_target": "False",
    "max_source_positions": max_source_positions,
    "max_target_positions": max_target_positions,
    "upsample_primary": 1,
    "truncate_source": False,
}
args["model_dir"] = model_dir


# In[3]:


# Toggle lite_mode to True to run a light version of the model.
# Considerable speedup on CPU with some impact on the diversity in paraphrases generated.
pp = NMTParaphraser(args, lite_mode=True)


# In[4]:


sentence = "I am searching for flights from New York to Berlin flying next week."
# sentence = "I am searching for apartments in a nice neighborhood."
# sentence = "I am hungry for some authentic Italian cuisine."
# sentence = "As soon as you think you have escaped the danger, you get hit by another one."
# sentence = "The president's announcement brought a sense of fear in the minds of people."


# In[5]:


prism_a = 0.05
prism_b = 4
paraphrases = pp.generate_paraphrase(sentence, "en", prism_a, prism_b)
for i, p in enumerate(paraphrases):
    print(f"{i+1}. {p}")


# In[6]:


sentence = "Ich möchte alle meine Karten sperren. Wie mache ich es?"
prism_a = 0.05
prism_b = 4
paraphrases = pp.generate_paraphrase(sentence, "de", prism_a, prism_b)
for i, p in enumerate(paraphrases):
    print(f"{i+1}. {p}")
