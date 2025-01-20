docker run --runtime nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm \
	-v /home/harish/cv/models/fall_detection_model_saved:/workspace/model \
	-v /home/harish/cv/models/fall_detection_model_rt:/workspace/output/tftrt_saved_model \
	-v /home/harish/cv/data/calibration_data:/workspace/data/calibration_data \
	tf-trt-converter --use_calibration=True --calibration_data=/workspace/data/calibration_data
