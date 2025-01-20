docker run --runtime nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm \
	-v /home/harish/cv/models/fall_detection_model_rt:/workspace/output/tftrt_saved_model \
	-v /home/harish/cv/data:/workspace/data \
	-v /home/harish/cv/app/test_fall_detection/output:/workspace/output \
	test-fall
