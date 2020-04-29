import os, sys, glob
import numpy as np
import pandas as pd

import argparse
import logging
import time
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

path = "/media/xin/New Volume/Large files/UCF-101"
model = "cmu"
tensorrt = False
resize_out_ratio = 4.0





def run_video(video, folder_name, id_value, visual=False):
	print(f"Running on {folder_name} Video ID: {id_value}")
	fps_time = 0
	w, h = model_wh('0x0')
	if w > 0 and h > 0:
		e = TfPoseEstimator(get_graph_path(model), target_size=(w, h), trt_bool=tensorrt)
	else:
		e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368), trt_bool=tensorrt)



	cam = cv2.VideoCapture(video)
	ret_val, image = cam.read()


	while True:
		ret_val, image = cam.read()
		if not ret_val:
			break
		humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
		print(humans)

		image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

		cv2.putText(image,
					"FPS: %f" % (1.0 / (time.time() - fps_time)),
					(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 0), 2)
		cv2.imshow('tf-pose-estimation result', image)
		fps_time = time.time()
		if cv2.waitKey(1) == 27:
			break


	cv2.destroyAllWindows()


def generate_row(humans):
	pass

#need to make sure 70% of frames contains body parts


def main():
	for folder in os.scandir(path):
		list_files = glob.glob(os.path.join(folder.path, r"*.avi"))
		print(f"running on {folder.path} data, {len(list_files)} files")
		for i, file in enumerate(list_files):
			run_video(file, os.path.basename(folder.path), i)
	




if __name__ == "__main__":
	#main()
	run_video("test.avi", "what", 1)


