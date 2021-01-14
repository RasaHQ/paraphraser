docker run --rm -it \
	-v $PWD/run_paraphraser.py:/home/run_paraphraser.py \
	-v $PWD/data:/etc/data \
	rasa/paraphraser:1.0.0 \
        python run_paraphraser.py \
	--language en --prism_a 0.05 \
	--input_file test.csv \
	--output_format csv
	# --interactive

