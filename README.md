# Vehicle Detection with Ensemble Learning

**Authors:** David Ornelas, José Santos, Raquel Pereira.  
**Affiliation:** IEETA/DETI, University of Aveiro, Aveiro, 3810-193, Portugal  

### Introduction
Repository content: Trained models with ensemble learning for vehicle detection using open source datasets, using YOLOv7 and YOLOv9.
Keep in mind the open-source datasets were not fully controlled, since several images contain missing labels, this could impact the model performance on out of the box images.
For further improvements, train these models on your dataset.    

### Installation:
To install the necessary dependencies for this project, follow the steps below.

1. **Clone the repository**:
	```bash
	git clone git@github.com:ornelas15/vehicle_detection.git or download .zip
	cd vehicle_detection
	```

2. **Install requirements**:
	```bash
	pip3 install -r requirements.txt
	```    


### Weights download:
To add both based (also available on original reps) and trained models.


### Model Training:

Before training, make sure to check out if all arguments are correct.
Required: 
- Weights
- CFG
- DATA
- HYP
- EPOCHS (20 def)
- BATCH_SIZE (16 def)
- IMG_SIZE (640 def)

To train on yolov7 use the following command, change the --args if needed:

1. **Run train.py for single GPU**:

	```bash
	python3 train.py --device 0 --data 'data/vehicle_dataset/data.yaml' --cfg 'cfg/yolov7.yaml' --weights 'weights/yolov7_training.pt' --hyp 'data/hyp.scratch.custom.yaml' --name yolov7-run1 --batch-size 16
	```

2. **Run train.py for dual GPU**:

	```bash
	python3 train.py --device 0,1 --sync-bn --data 'data/vehicle_dataset/data.yaml' --cfg 'cfg/yolov7.yaml' --weights 'weights/yolov7_training.pt' --hyp 'data/hyp.scratch.custom.yaml' --name yolov7-run1 --batch-size 16
	```
  
	To train on yolov9 use the following command, change the --args if needed:

1. **Run train.py for single GPU**:

	```bash
	python3 train.py --device 0 --batch 16 --data 'data/vehicle_dataset/data.yaml' --cfg 'models/detect/yolov9-c.yaml' --weights 'weights/yolov9-c.pt' --name yolov9-run1 --hyp 'data/hyps/hyp.scratch-high.yaml'
	```

2. **Run train.py for dual GPU**:

	```bash
	python3 train_dual.py --device 0,1 --sync_bn --batch 16 --data 'data/vehicle_dataset/data.yaml' --cfg models/detect/yolov9-c.yaml --weights 'weights/yolov9-c.pt' --name yolov9-run1 --hyp 'data/hyps/hyp.scratch-high.yaml'   
	```


### Model Testing:
Before testing, make sure to check out if all arguments are correct.

1. **Test YOLOv7 trained**:

	```bash
	python3 test.py --weights './yolov7-best.pt'
	```
	
	Run test.py with --device 0,1 and --sync_bn if dual GPU.
  
2. **Test YOLOv9 trained**:

	```bash
	python3 val.py --conf 0.001 --iou 0.7 --weights './yolov9c-best.pt' 
	```

	Run val_dual with --device 0,1 and --sync_bn if dual GPU.  
     

### Inference:

1. **Inference on images with YOLOv7:**
	```bash
	python3 detect.py --source './data/image.jpg' --weights './yolov7-best.pt' --name yolov7_inference
	```
	Run detect.py with --device 0,1 and --sync_bn if dual GPU.
     
1. **Inference on images with YOLOv9:**

	```bash
	python3 detect.py --source './data/image.jpg' --weights './yolov9c-best.pt' --name yolov9_inference
	```
	Run detect_dual with --device 0,1 and --sync_bn if dual GPU.  


### Ensemble Learning (WBF):

1. **Apply WBF on trained weights**:
	```bash
	python3 wbf_eval.py
	```

2. **View WBF**: 
	```bash
	python3 wbf_view.py
	```  


### References
This code was adapted from the original versions of the YOLOv7 and YOLOV9 official repository in the following links:

[YOLOv7 GitHub](https://github.com/WongKinYiu/yolov7)  
[YOLOv9 GitHub](https://github.com/WongKinYiu/yolov9)  
  
	
YOLO articles:


	@inproceedings{wang2023yolov7,
	  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
	  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year={2023}
	}

	@article{wang2024yolov9,
	  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
	  author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
	  booktitle={arXiv preprint arXiv:2402.13616},
	  year={2024}
	}

	



