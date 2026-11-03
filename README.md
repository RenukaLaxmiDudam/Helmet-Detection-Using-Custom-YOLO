# Helmet Detection Using Custom YOLO

## 1. Project overview
A custom-trained **YOLOv8** object detector that finds **helmets worn by two-wheeler riders** in images and videos. It draws bounding boxes with the class name and confidence score, and ships with a **Streamlit** web app.

## 2. Problem statement
Riding without a helmet is a major cause of fatal two-wheeler accidents. Manual monitoring is slow and error-prone. This project builds an automatic, camera-based helmet detector as a first step towards helmet-compliance monitoring.

## 3. Objectives
- Train a custom YOLO model (transfer learning) to detect helmets.
- Evaluate it on a held-out test set with standard metrics.
- Run detection on images and videos.
- Provide a Streamlit app for easy use and deployment.

## 4. Technologies used
Python, Google Colab, PyTorch, Ultralytics YOLOv8, OpenCV, Pillow, NumPy, Pandas, Matplotlib, Streamlit, GitHub.

## 5. Dataset
- **Source:** Roboflow Universe, project `helmet-motorcycle-detection-pro0q` (workspace `computer-vision-vxgzc`), version 1. https://universe.roboflow.com/computer-vision-vxgzc/helmet-motorcycle-detection-pro0q  
- **License:** as stated in the dataset's `data.yaml` / README (Public Domain at the time of download).
- **Format:** YOLOv8 (one `.txt` label file per image).
- **Classes:** `Helmet` (ID 0)
- **Important limitation:** the dataset has only the class *Helmet*. The model detects helmets; it **cannot** detect riders *without* helmets. That needs extra, properly labelled "No Helmet" data.
- Roboflow pre-processing: auto-orient, resize to 640x640 (stretch). Training images were augmented by Roboflow (flip, 90-degree rotation, blur, noise), creating about 3 copies per source photo.

## 6. Dataset splitting
The downloaded ZIP contained 2931 images, split by Roboflow as train: 2703, val: 114, test: 114. It was not 70:20:10 (and some source photos appeared in more than one split), so a new reproducible split (seed 42, grouped by source photo) was created.

Final split used for training/evaluation (target 70% / 20% / 10%):

| Split | Images | Helmet boxes |
|---|---|---|
| train | 2043 | 7956 |
| val | 594 | 2257 |
| test | 294 | 1120 |

## 7. YOLO architecture
YOLOv8n (nano) from Ultralytics: a CSP-style convolutional backbone for feature extraction, a PAN/FPN neck that merges features from several scales, and an anchor-free decoupled head that predicts box position and class separately. The model has about 3,138,003 parameters. Training started from COCO-pretrained `yolov8n.pt` (transfer learning).

## 8. Training configuration
| Setting | Value |
|---|---|
| Base weights | yolov8n.pt (pretrained) |
| Max epochs | 50 (early stopping patience: 15) |
| Epochs actually trained | 50 |
| Image size | 640 |
| Batch size | 32 |
| Optimizer | auto |
| Workers | 2 |
| Ultralytics version | 8.4.199 |
| Project / run name | Helmet_Detection / custom_yolov8 |

## 9. Evaluation metrics
- **Precision** - of the boxes drawn, the fraction that are real helmets.
- **Recall** - of the real helmets, the fraction found.
- **mAP50** - mean average precision at IoU 0.50.
- **mAP50-95** - mean average precision averaged over IoU 0.50 to 0.95 (stricter).

## 10. Results
Values below were produced by the notebook (Section 10). Precision and recall are reported at the confidence threshold that maximizes F1.

| Metric | Test set | Validation set |
|---|---|---|
| Precision | 0.9431 | 0.9572 |
| Recall | 0.9777 | 0.9512 |
| F1 | 0.9601 | 0.9542 |
| mAP50 | 0.9884 | 0.9883 |
| mAP50-95 | 0.9433 | 0.9388 |

Plots (confusion matrix, PR curve, F1 curve, training curves) are in the `screenshots/` folder.

*Caveat:* validation and test images include Roboflow-augmented copies, and the dataset is small, so real-world performance on new cameras, lighting and angles may be lower than these numbers.

## 11. How to run locally
```bash
git clone <YOUR-GITHUB-REPO-URL>
cd Helmet-Detection-Using-Custom-YOLO
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Command-line detection:
```bash
python inference/image_detection.py --source path/to/image.jpg --conf 0.25
python inference/video_detection.py --source path/to/video.mp4 --conf 0.25
```
Retrain (needs the dataset and a `data.yaml`): `python training/train.py --data path/to/data.yaml`

## 12. Streamlit Community Cloud deployment
1. Push this folder to a **public** GitHub repository (include `best.pt`, `app.py`, `requirements.txt`, `packages.txt`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository and branch, set the main file path to `app.py`.
4. Click **Deploy** and wait for the build (first build takes a few minutes).

## 13. Links (placeholders - not deployed yet)
- GitHub repository: `<ADD YOUR GITHUB REPOSITORY LINK HERE>`
- Streamlit application: `<ADD YOUR STREAMLIT APP LINK HERE AFTER DEPLOYMENT>`
- Hugging Face Space: `<ADD YOUR HUGGING FACE SPACE LINK HERE IF YOU DEPLOY THERE>`

## 14. Conclusion
A one-class YOLOv8n helmet detector was trained with transfer learning and evaluated on a held-out test split (see Results above). The app can detect helmets in images and videos. Future work: add a labelled "No Helmet" class, add number-plate recognition for enforcement, collect more varied data (night, rain, different cities), and test on real traffic video.
