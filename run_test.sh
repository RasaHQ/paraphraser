docker run --rm -it \
	-v $PWD/run_paraphraser.py:/home/run_paraphraser.py \
	rasa/paraphraser:1.0.0 \
        python run_paraphraser.py --interactive --language en --prism_a 0.05
