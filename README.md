YOLOv8_MSG adalah YOLOv8 versi nano yang telah dioptimasi menggunakan metode:
- Implementasi <a href="https://arxiv.org/abs/1911.11907">Ghost Module</a>
  - Modifikasi Bagian Head (Detect): Detect_GhostModule (ultralytics/nn/modules/head.py)
- Penambahan Proses Down-sampling
  - Modifikasi Conv Module: Conv_Avg_Pooling (ultralytics/nn/modules/conv.py)
- Integrasi Attention Mechanism (<a href="https://arxiv.org/abs/1807.06521">CBAM</a>)
  - Modifikasi C2f Module: CBAM_C2f (ultralytics/nn/modules/block.py)

## <div align="center">Dokumentasi</div>

## Cara Intall & Contoh Penggunaan

<summary>Cara Install</summary>
<br>

```bash
git clone https://github.com/MInsanKamil/YOLOv8_MSG.git
```

<summary>Contoh Penggunaan</summary>

### Python

- Nama Model:
  - yolov8n_GhostModule.yaml (YOLOv8n + Ghost Module)(ultralytics/cfg/models/v8/yolov8n_GhostModule.yaml)
  - yolov8n_GhostModule_Avg_Pooling.yaml (YOLOv8n + Ghost Module + Down-sampling)(ultralytics/cfg/models/v8/yolov8n_GhostModule_Avg_Pooling.yaml)
  - yolov8n_GhostModule_Avg_Pooling_CBAM.yaml (YOLOv8n + Ghost Module + Down-sampling + Attention Mechanism)(ultralytics/cfg/models/v8/yolov8n_GhostModule_Avg_Pooling_CBAM.yaml)

```bash
cd YOLOv8_MSG
```

```python
from ultralytics.models.yolo.model import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v8/nama_model.yaml") # build a new model from scratch
model = YOLO("ultralytics/cfg/models/v8/nama_model.yaml").load("yolov8n.pt")  # load weight pretrained yolov8n coco dataset

# Load a model pretrained yolov8_msg indoor dataset
model = YOLO('best.pt')

# Use the model
model.train(data="coco8.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

### Notebooks

| Notebook                                                                                                                           | 
| <a href="https://colab.research.google.com/drive/1Lx3UP3TE2dDNbCZMLID4DaG7uIWnninW#scrollTo=75-VWNhGmS9q">Evaluasi Model</a>                                      | 

## <div align="center">Models</div>

Dibawah ini hasi  evaluasi model untuk mendeteksi objek dalam rumah ([Indoor Object Dataset](https://app.roboflow.com/csgitk/indoor_object_ta/10)) 

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50 | mAP<sup>test<br>50 | GFLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | -------------------- | ----------------- |
| YOLOv8n (Baseline) | 640                   | 74.2                 | 72                | 8.09               |
| YOLOv8n + Ghost Module | 640                   | 73.3                | 73.1                | 6.75               |
| YOLOv8n + Ghost Module + Avg Pooling| 640                   | 72.7                | 72.5               | 1.77               |
| YOLOv8n + Ghost Module + Avg Pooling + CBAM| 640                   | 74.4                | 73.6               | 1.84               |
