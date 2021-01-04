import argparse
import os
import time
import logging
logger = logging.getLogger()

from paraphraser.modelling.nmt_paraphraser import NMTParaphraser


def run_interactive(model, args):

    logger.info(f"Starting paraphraser in interactive mode. Language set to {args.language}")

    while True:
        input_sentence = input("Enter a sentence to be paraphrased: ")
        start = time.time()
        paraphrases = model.generate_paraphrase(input_sentence, args.language, args.prism_a, args.prism_b)
        end = time.time()

        for i, p in enumerate(paraphrases):
            print(f"{i + 1}. {p}")

        print(f"Generation completed in {end-start}")


def run_bulk(model, args):
    pass


def run(args):

    validate_args(args)

    model = NMTParaphraser(run_args=args, lite_mode=args.lite)

    if args.interactive:
        run_interactive(model, args)
    else:
        run_bulk(model, args)


def validate_args(args):

    if not args.interactive and not args.input_file:
        raise ValueError("You chose to run the paraphraser in bulk mode but did not specify a "
                         "file to pick up the input sentences from. Either run in interactive "
                         "mode(--interactive) or pass the path to a file to be paraphrased with "
                         "--input_file option")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--lite", action='store_true', help="Run paraphraser in lite mode, i.e. do not apply "
                                                          "downweight penalty during decoding. "
                                                          "This results in speedup at the cost of "
                                                          "diversity of generated paraphrases.")
    parser.add_argument("--interactive", action="store_true", help="Run paraphraser interactively in the shell.")

    parser.add_argument("--language", required=True, help="Language code corresponding to the language of input sentence.")
    parser.add_argument("--input_file", default="", help="File containing input sentences to be paraphrased")

    parser.add_argument("--prism_a", type=float, default=0.0)
    parser.add_argument("--prism_b", type=int, default=4)
    parser.add_argument("--diverse_beam_groups", type=int, default=10)
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--nbest", type=int, default=10)
    parser.add_argument("--diverse_beam_strength", type=float, default=0.5)
    parser.add_argument("--max_source_positions", type=int, default=50)
    parser.add_argument("--max_target_positions", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1)

    args = parser.parse_args()

    run(args)
