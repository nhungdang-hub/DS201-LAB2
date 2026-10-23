# Bài thực hành 2 — Mạng Neural tích chập

Project hoàn thành Bài 2 (GoogLeNet), Bài 3 (ResNet-18) và Bài 4
(Hugging Face ResNet-50) trên VinaFood21. Source Bài 1 trong `Mẫu/Mẫu/`
được giữ nguyên.

## 1. Phân tích bài mẫu

Bài 1 tổ chức theo luồng đơn giản:

1. `MnistDataset` đọc toàn bộ file IDX, chuẩn hóa pixel và trả dictionary
   `{"image", "label"}`; `collate_fn` ghép các mẫu thành batch.
2. `LeNet` tự định nghĩa các lớp convolution, average pooling và fully connected;
   `forward()` trả logits, không Softmax trước loss.
3. Script dùng `CrossEntropyLoss`, Adam và vòng lặp
   `zero_grad → forward → loss → backward → step`.
4. Evaluation chuyển logits thành nhãn bằng `argmax`, tính accuracy, precision,
   recall và F1-macro, rồi lưu `state_dict` tốt nhất theo F1.

Hạn chế quan trọng của code mẫu là evaluation gọi `.item()` cho cả batch nên chỉ
chạy đúng khi batch size bằng 1; đồng thời tập test đang được dùng để chọn
checkpoint. Engine của Bài 2–4 gom tensor theo batch và chọn checkpoint **chỉ từ
validation macro-F1**.

## 2. Cấu trúc project

```text
.
├── assignments (6).ipynb       # đề chính thức, không sửa
├── Mẫu/Mẫu/                     # Bài 1, không sửa
├── assignment-2.py              # train GoogLeNet
├── assignment-3.py              # train ResNet-18
├── assignment-4.py              # fine-tune/inference ResNet-50
├── data_utils/
│   └── vinafood21_dataset.py
├── models/
│   ├── googlenet.py
│   └── resnet18.py
├── training/
│   ├── engine.py
│   └── metrics.py
└── tests/test_models.py
```

Checkpoint và lịch sử được ghi vào `checkpoints/assignment-{2,3,4}/`.
Mỗi lần train tạo `best_model.pt`, `history.json` và, khi có matplotlib,
`learning_curves.png`.

## 3. Kiến trúc

### GoogLeNet / Inception v1

`InceptionBlock` chạy song song bốn nhánh: convolution 1×1; 1×1→3×3;
1×1→5×5; và MaxPool 3×3→1×1. Kết quả được nối theo chiều channel. Model có đủ
3a, 3b, 4a–4e và 5a–5b với output channel lần lượt
`256, 480, 512, 512, 512, 528, 832, 832, 1024`. Convolution đầu là
7×7/stride 2/padding 3; mọi MaxPool bật `ceil_mode=True`. Cuối mạng là global
average pooling, dropout 0.4 và linear 21 lớp. Không có auxiliary classifier và
không dùng pretrained weights.

### ResNet-18 và residual connection

Mỗi `BasicBlock` có hai convolution 3×3, BatchNorm và residual addition. Shortcut
là identity khi số channel giữ nguyên; khi đổi channel, projection convolution
1×1 làm hai vế có cùng shape. Bốn stage có `[2, 2, 2, 2]` block và
`[64, 128, 256, 512]` channel.

Notebook yêu cầu MaxPool `kernel=3, stride=2, padding=0` **giữa** các residual
stage. Project diễn giải đây là ba pool ở ranh giới conv2_x→conv3_x,
conv3_x→conv4_x, conv4_x→conv5_x và đặt stride của block bằng 1, tránh downsample
hai lần. Vì padding bằng 0, kích thước thực là `56→27→13→6`, không thể đồng thời
khớp bảng ResNet chuẩn `56→28→14→7` (bảng chuẩn downsample bằng stride 2 trong
block). Đây là sai khác có chủ ý để giữ đúng yêu cầu chữ của notebook; không đặt
MaxPool sau từng BasicBlock.

### Pretrained ResNet-50

Bài 4 nạp thật checkpoint `microsoft/resnet-50` bằng
`AutoModelForImageClassification.from_pretrained`, đặt 21 nhãn và dùng
`ignore_mismatched_sizes=True` chỉ để thay classification head. Fine-tuning gồm:

- Giai đoạn 1: đóng băng toàn bộ `model.resnet`, chỉ train classifier.
- Giai đoạn 2: mở stage cuối của backbone; backbone dùng LR nhỏ, head dùng LR lớn;
  embedder và các stage còn đóng băng tiếp tục ở `eval()`.

Trong giai đoạn 1, backbone được giữ ở `eval()` để BatchNorm không âm thầm cập
nhật running statistics khi weights đã bị đóng băng.

Model tốt nhất, `label2id`/`id2label` và `AutoImageProcessor` được lưu theo định
dạng Hugging Face để nạp lại inference. Processor tự resize/crop/normalize theo
checkpoint; augmentation train xảy ra trên ảnh PIL trước processor nên không có
normalize hai lần.

## 4. Chuẩn bị VinaFood21

Dataset thật đã được giải nén tại `dataset/VinaFood21/`: train có 10.044 ảnh và
test có 6.682 ảnh, mỗi split đủ 21 lớp. Dataset gốc không có validation split,
vì vậy loader tách validation có phân tầng từ train; test chỉ dùng để đánh giá
sau khi đã chọn checkpoint.

Loader kiểm tra dữ liệu trên đĩa trước khi dùng và hỗ trợ:

```text
VinaFood21/                  VinaFood21/
├── train/                   ├── class_a/
│   ├── class_a/             ├── class_b/
│   └── ...                  └── ...
├── validation/ (hoặc val/)
└── test/
```

Ngoài ra có thể dùng `train.csv`, `validation.csv`/`val.csv`, `test.csv` với một
cột đường dẫn (`path`, `image_path`, `filename`, ...) và một cột nhãn (`label`,
`class_name`, ...). Nếu chỉ có các thư mục lớp hoặc không có validation, loader
tách train/validation có phân tầng với seed cố định. Mapping lớp là thứ tự tên lớp
đã sort trên hợp của mọi split và được dùng chung. Chương trình dừng nếu không
phát hiện đúng 21 lớp, đồng thời in số ảnh theo từng lớp/split trước khi train.

## 5. Cài đặt và chạy

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Chạy từng bài:

```powershell
python assignment-2.py --data-root dataset/VinaFood21 --epochs 30 --batch-size 32
python assignment-3.py --data-root dataset/VinaFood21 --epochs 30 --batch-size 32
python assignment-4.py --data-root dataset/VinaFood21 --head-epochs 5 --epochs 10
```

Inference ảnh mới sau Bài 4:

```powershell
python assignment-4.py --inference-image path/to/image.jpg
```

Các CLI đều có `--learning-rate`, `--seed`, `--device` (`auto`, `cpu`, `cuda`),
`--num-workers` và `--validation-ratio`. Bài 4 thêm `--head-learning-rate`.
Giá trị mặc định chính: Adam, batch 32, seed 42, LR 1e-3 cho model train từ đầu;
Bài 4 dùng backbone LR 1e-4 và head LR 1e-3. Trên Windows, `num_workers=0` là
mặc định an toàn và có thể tăng khi máy cho phép.

Chạy test:

```powershell
python -m pytest tests/test_models.py -vv
```

## 6. Chạy trên Kaggle

Không đưa `dataset/` hoặc checkpoint lên GitHub; hai thư mục này đã nằm trong
`.gitignore`. Trên Kaggle, thêm VinaFood21 qua mục **Input**, bật GPU trong
**Settings → Accelerator**, clone repository vào `/kaggle/working` và truyền
đường dẫn dataset thực tế:

```bash
git clone <URL_REPOSITORY>
cd <TEN_REPOSITORY>
pip install -r requirements.txt
python assignment-2.py --data-root /kaggle/input/<dataset-slug>/VinaFood21 --num-workers 4
python assignment-3.py --data-root /kaggle/input/<dataset-slug>/VinaFood21 --num-workers 4
python assignment-4.py --data-root /kaggle/input/<dataset-slug>/VinaFood21 --num-workers 4
```

Bài 4 cần Internet ở lần tải đầu tiên của `microsoft/resnet-50`, hoặc checkpoint
Hugging Face phải được thêm sẵn dưới dạng Kaggle Input. Checkpoint huấn luyện được
ghi vào `/kaggle/working/<TEN_REPOSITORY>/checkpoints/` để có thể tải về sau run.

## 7. Kết quả thực nghiệm

### Bài 1 — LeNet trên MNIST

Đã chạy đủ 10 epoch trên CPU với batch size 32. Thời gian huấn luyện và đánh giá
khoảng 6 phút; checkpoint tốt nhất được lưu tại
`checkpoints/assignment-1/best_model.pth`.

| Accuracy | Precision macro | Recall macro | F1 macro |
|---:|---:|---:|---:|
| 98.46% | 98.43% | 98.47% | 98.44% |

Checkpoint tốt nhất đã được nạp lại và đánh giá trên toàn bộ 10.000 ảnh test.
Code mẫu dùng chính tập test để chọn checkpoint nên kết quả này chỉ dùng để báo
cáo Bài 1, không so sánh trực tiếp với các mô hình VinaFood21 bên dưới.

### Bài 2–4 — VinaFood21

Chỉ điền số liệu sau khi mô hình đã được huấn luyện trên dữ liệu thật.

| Model | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| GoogLeNet (scratch) | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| ResNet-18 (scratch) | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| ResNet-50 pretrained | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |

Sau khi train, lấy metric test được in bởi script hoặc từ log thực tế để cập nhật
bảng và chỉ so sánh ba model khi cả ba đã được đánh giá trên cùng test split.
Validation dùng để chọn model; test chỉ được gọi sau khi checkpoint tốt nhất đã
được nạp lại.

## 8. Trạng thái kiểm thử hiện tại

Trên môi trường CPU hiện tại: 9 test đã pass (forward/shape GoogLeNet và
ResNet-18, channel Inception, 8 BasicBlock, pooling riêng, projection shortcut,
metrics với batch lớn hơn 1, một bước forward/backward/Adam và việc giữ BatchNorm
của backbone đóng băng). Engine còn được kiểm tra với output kiểu Hugging Face
`.logits` và trường hợp phase mới không cải thiện thì phải khôi phục checkpoint
baseline tốt hơn. Test tải weights Hugging Face được skip vì checkpoint chưa có
trong cache. MNIST thật đã qua một bước forward/backward; VinaFood21 thật đã qua
kiểm tra discovery, stratified split và batch PIL ở cả train/validation/test.
Full training chưa được chạy và không có metric thực nghiệm giả.
