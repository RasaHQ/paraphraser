docker run --rm -it \
	-v $PWD/run_paraphraser.py:/home/run_paraphraser.py \
	-v $PWD/data:/etc/data \
	dakshvar22/paraphraser_cpu \
        python run_paraphraser.py \
	--language en --prism_a 0.05 \
	--input_file dummy.yaml \
	--output_format yaml \
	--lite
	# --interactive

