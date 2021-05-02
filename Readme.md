## Paraphraser

<img src="imgs/square-logo.svg" width=200 height=200 align="right">

Paraphraser is a tool that you can use to generate paraphrases of sentences. The generated paraphrases can be expected to be semantically similar to the original sentence but differ lexically and structurally.

For example -

Input sentence: `Go through the green door, but don't walk across the red tiles.`

Generated paraphrases:
```
- Go through the green door, but don't walk through the red tiles.
- Leave it through the green door, but don't walk through the red tiles.
- walking through the green door, but not the red tiles.
- Take the green door but don't walk through the red tiles.
- go across Green Door, but don't walk through the REDs.
- Running through this "green door", but not walking across these "red tiles".
- Go through the green door, but don't walk over the red tiles.
```

It uses a transformer based natural language generation model which has been trained on a large text corpus mined from the web. It supports 30 different languages.(See model details below for more information).
The tool was primarily built as part of building a data augmentation pipeline inside the Rasa Open Source framework but you can use it for generating paraphrases for any of your use case as well.

Feedback is welcome!

**Warning: Please be aware that since the underlying model is trained on a large corpus from the web, it can generate offensive content too. The tool doesn't filter/detect any such content as of now.**

## Docker setup

You will need to install docker engine to use the tool ([installation guide](https://docs.docker.com/engine/install/ubuntu/)). The docker setup has been tested with docker version `20.10.1`. You can check the installed version with `docker --version`.

We host the docker images on dockerhub as well, but if you are interested in playing with the model extensively we recommend you to build the docker image from scratch -

First, clone the repository:

```bash
git clone https://github.com/RasaHQ/paraphraser.git
```

Navigate into the `paraphraser` directory and follow the steps below depending on your hardware setup, i.e. CPU v/s GPU setup.

### CPU

Build the image with -

```bash
docker build -f Dockerfile.cpu -t paraphraser_cpu:latest .
```

### GPU

You will need to install `nvidia-docker` as well to run the tool on GPU enabled devices. Refer to guides like [this](https://gist.github.com/nathzi1505/d2aab27ff93a3a9d82dada1336c45041) to do so.

Build the image with -

```bash
docker build -f Dockerfile.gpu -t paraphraser_gpu:latest .
```

Next, when running the tool (commands in usage section below) use the appropriate image name depending on CPU or GPU mode and additionally append `--gpus all` if running on GPU.

### Usage

There are two modes in which you can run the tool -

#### Interactive mode

As the name suggests, this mode lets you generate paraphrases in an online setting on the CLI. To start the tool in this mode -

```bash
docker run --rm -it \
    dakshvar22/paraphraser_cpu:latest \
    --interactive \
    --language en
```

Use `dakshvar22/paraphraser_gpu:latest` as docker image name and add `--gpus all` flag to the `docker run` command if running on GPU, for e.g. -

```bash
docker run --gpus all --rm -it \
    dakshvar22/paraphraser_gpu:latest \
    --interactive \
    --language en
```

#### Bulk Generation mode

This mode lets you run the tool on collection of sentences in bulk. There are two input formats we currently support -

1. Rasa's [NLU training data format](https://rasa.com/docs/rasa/training-data-format/#nlu-training-data).
2. CSV format with each line being - `<sentence>, <optional-label>`

You can generate the output in two formats -

1. Rasa's NLU Training data format(recommended) - The paraphrases for each example will be added as metadata of that example. For example, if the original data looked like -
```
nlu:
- intent: ask_query
  examples:
   - How to apply for passport

```
The generated paraphrases would be appended in this manner -
```yaml
nlu:
- intent: ask_query
  examples:
   - text: How can I apply for a new passport?
     metadata: 
       paraphrases: |
       - How to apply for passport
       - Apply for a new passport.
       scores: |
       - 0.82
       - 0.93
```

The `scores` section of the yaml shows the semantic similarity of each paraphrase with the original sentence. This is computed with the [multi-lingual USE model](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)

2. CSV Format - Each paraphrase will be added in a separate line as = `<original-sentence>, <optional-label>, <paraphrase>`.

To generate paraphrases in this bulk mode, run -

```bash
docker run --rm -it \
    -v $PWD/data:/etc/data \
    dakshvar22/paraphraser_cpu:latest \
    --input_file test.yaml \
    --output_format yaml \
    --language en
```

Use `dakshvar22/paraphraser_gpu:latest` as docker image name and add `--gpus all` flag to the `docker run` command if running on GPU, for e.g. -

```bash
docker run --gpus all --rm -it \
    -v $PWD/data:/etc/data \
    dakshvar22/paraphraser_gpu:latest \
    --input_file test.yaml \
    --output_format yaml \
    --language en
```

Also note that the path to input file should be relative to the directory that you are mounting to `/etc/data` which is `$PWD/data` here.

**We suggest you to try out the tool in interactive mode first before trying out the bulk generation mode in order to understand which parameters of the model suit your data the best.**

## Model details

We use the same multi-lingual neural machine translation(NMT) model used by the [Prism](https://github.com/thompsonb/prism) [1][2] framework. The paraphrases are generated by applying a diversity promoting penalty term to the decoding strategy used by the original model. Please refer to the [original paper](https://arxiv.org/abs/2008.04935) for more details. We made a small modification on top of it to make a lighter version of the model by [artifically boosting the log probability of predicted tokens](src/nmt_paraphraser/utils.py#7) and using [diverse beam search](https://arxiv.org/abs/1610.02424) for decoding more diverse paraphrases across multiple generations. Since, the underlying NMT model is a multi-lingual one, it can generate paraphrases in all the languages it was original trained for -

Albanian (sq), Arabic (ar), Bengali (bn), Bulgarian (bg), Catalan Valencian (ca), Chinese (zh), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), English (en), Esperanto (eo), Estonian (et), Finnish (fi), French (fr), German (de), Greek, Modern (el), Hebrew (modern) (he), Hungarian (hu), Indonesian (id), Italian (it), Japanese (ja), Kazakh (kk), Latvian (lv), Lithuanian (lt), Macedonian (mk), Norwegian (no), Polish (pl), Portuguese (pt), Romanian, Moldavan (ro), Russian (ru), Serbian (sr), Slovak (sk), Slovene (sl), Spanish; Castilian (es), Swedish (sv), Turkish (tr), Ukrainian (uk), Vietnamese (vi)

However, we have only tested the tool currently for English (en), German (de), French (fr), Spanish (es) and Russian (ru). We encourage users to try out the tool in other languages and share their results!

## Additional parameters to try

These are some additional parameters that you can append to the above commands while running the tool.

### Model Parameters

1. `--lite`: Use this option to run the lightweight version of the tool. The lightweight version will be around 2.5x faster but may also negatively affect the generation quality too.
2. `--language`: Language code, for e.g. - `en` for English.

### Parameters affecting generation

1. `--prism_a`: Controls the trade-off between semantic similarity and lexical diversity. Set it in the range `0.001 - 0.1`, where higher values promote more lexical diversity. Default - `0.5`.
2. `--prism_b`: Maximum n while computing n-grams for penalization during generation. Set it in range `2 - 4`, where higher values will penalize higher order n-grams. Default - `4`.
3. `--diverse_beam_groups`: Number of groups for diverse beam search. Usually set to the number of paraphrases you want to generate for an example. Default - `10`. Higher values will affect the inference speed.
4. `--beam`: Beam size for diverse beam serach. Usually set it equal to the value of `diverse_beam_search`.
5. `--nbest`: Number of candidates of the beam to carry forward in every timestep. Usually set it equal to the value of `diverse_beam_search` and number of paraphrases you want to generate in the end.

### Parameters for I/O

1. `--interactive`: Use to run the tool in interactive mode.
2. `--input_file`: Path to input file containing sentences to be paraphrased. Used only in bulk mode.
3. `--output_format`: File format for generated paraphrases. `yaml`, `yml`, `csv`.


## References

<a id="1">[1]</a>
Thompson et. al. (2020).
Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing,
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)

<a id="2">[2]</a>
Thompson et. al. (2020).
Paraphrase Generation as Zero-Shot Multilingual Translation: Disentangling Semantic Similarity from Lexical and Syntactic Diversity,
Proceedings of the Fifth Conference on Machine Translation (Volume 1: Research Papers)
