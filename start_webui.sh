#!/bin/bash

set -ex


CUDA_VISIBLE_DEVICES=0 python gradio_annotator.py &
CUDA_VISIBLE_DEVICES=1 python gradio_canny2image.py &
CUDA_VISIBLE_DEVICES=2 python gradio_depth2image.py &
CUDA_VISIBLE_DEVICES=3 python gradio_fake_scribble2image.py &
CUDA_VISIBLE_DEVICES=4 python gradio_hed2image.py &
CUDA_VISIBLE_DEVICES=5 python gradio_hough2image.py &
CUDA_VISIBLE_DEVICES=6 python gradio_normal2image.py &
CUDA_VISIBLE_DEVICES=7 python gradio_pose2image.py &
CUDA_VISIBLE_DEVICES=1 python gradio_scribble2image.py &
CUDA_VISIBLE_DEVICES=2 python gradio_scribble2image_interactive.py &
CUDA_VISIBLE_DEVICES=3 python gradio_seg2image.py &


#---------------------对应端口地址---------------------
#annotator	http://202.168.100.165:7860
#canny2image	http://202.168.100.165:7861
#depth2image	http://202.168.100.165:7862
#fake_scribble2image	http://202.168.100.165:7863
#hed2image	http://202.168.100.165:7864
#hough2image	http://202.168.100.165:7865
#normal2image	http://202.168.100.165:7866
#pose2image	http://202.168.100.165:7867
#scribble2image	http://202.168.100.165:7868
#scribble2image_interactive	http://202.168.100.165:7869
#seg2image	http://202.168.100.165:7870
#---------------------------------------------------


#杀掉
#ps -ef|grep python|grep gradio_|awk -F ' ' '{print $2}'|xargs -i kill -15 {}