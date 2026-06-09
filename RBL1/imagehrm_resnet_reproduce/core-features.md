# Ghi chú tái lập ImageHRM ResNet

Paper: "Hierarchical and Tiny Recursive Models for Medical Image Captioning"

PDF nguồn: https://rxiv.org/pdf/2601.0081v1.pdf

## Mục tiêu tái lập

Folder này tái lập nhánh ResNet của paper, không tái lập Swin hoặc FuseLIP.

Contract chính:

- Task: medical image captioning.
- Dataset: ROCOv2.
- Backbone: ResNet18 frozen.
- Kích thước ảnh: 224 x 224.
- Độ dài text: 512 token.
- Tokenization: ASCII/BPE theo mô tả của paper.
- Reasoning model: ImageHRM với visual-textual merge.
- Training: AdamW trong 50 epochs.
- Evaluation: captioning metrics, đặc biệt ROUGE-L và CIDEr.

## Ý tưởng lõi

Paper cho rằng medical image captioning không chỉ là detect vật thể rồi sinh câu trôi chảy. Một caption/report X-quang tốt cần đi qua ba tầng suy luận:

1. Quan sát thị giác cấp thấp.
2. Gom cụm findings theo anatomy/pathology.
3. Impression cấp cao.

Vì vậy model thêm latent recurrent reasoning trước khi dự đoán token. Thay vì dùng decoder fixed-depth, model cập nhật nhiều trạng thái ẩn rồi dùng trạng thái đó để sinh caption.

## Đường ResNet

Nhánh ResNet trong notebook đi theo đúng flow paper mô tả:

1. Load image.
2. Resize về 224 x 224.
3. Normalize theo ImageNet.
4. Chạy qua ResNet18 dùng ImageNet weights.
5. Bỏ classification head.
6. Freeze ResNet weights.
7. Lấy global average pooled feature vector 512 chiều.
8. Project visual vector sang cùng chiều với token embedding.

Công thức method section:

```text
Ev = Wv V
Xt = Et + Ev
```

Trong đó:

- `V`: visual feature từ ResNet18.
- `Ev`: visual embedding sau projection.
- `Et`: token embedding của caption.
- `Xt`: input đã merge image-text tại mỗi token step.

Đây là merge architecture: visual context được cộng vào mọi token position, không chỉ dùng để khởi tạo hidden state một lần.

## Đường HRM

Paper mở rộng dual-loop HRM thành triple-loop H-M-L:

- `H`: high-level planning, tương ứng global diagnosis/impression.
- `M`: middle semantic clustering, tương ứng gom findings theo anatomy/pathology.
- `L`: low-level syntax/token generation.

Pseudocode trong paper:

```text
for each H-cycle:
    for each M-cycle:
        for each L-cycle:
            update zL based on zL and Input
        update zM based on zM and zL
    update zH based on zH and zM
```

Notebook implement trực tiếp cho biến thể ResNet:

- `zL` nhận merged token-image features.
- `zM` nhận trạng thái `zL`.
- `zH` nhận trạng thái `zM`.
- Token logits sinh từ recurrent state đã kết hợp.

## Baseline cần so

Paper dùng `ResNet+LSTM` làm fixed-depth baseline. Notebook có baseline đơn giản với cùng ResNet18 frozen image path và LSTM decoder để giữ đúng hướng so sánh:

- Fixed-depth baseline: `ResNetLSTMBaseline`.
- Recurrent reasoning model: `ImageHRMResNetCaptioner`.

## Kết quả ResNet trong paper

Các dòng ResNet được report trên ROCOv2:

| Model | Backbone | H/M/L Config | ACT Loss | ROUGE-L | CIDEr |
| --- | --- | --- | ---: | ---: | ---: |
| ResNet+LSTM | ResNet18 | N/A | 1.87 | 0.106 | 0.310 |
| ImageHRM Dual | ResNet18 | 1/0/1 | 0.53 | 0.125 | 0.420 |
| ImageHRM Triple | ResNet18 | 1/1/1 | 0.49 | 0.157 | 0.478 |
| ImageTRM ResNet | ResNet18 | tiny recursive | N/A | 0.191 | 0.388 |

Notebook tập trung vào ba dòng đầu vì cùng dùng ResNet18 image encoder và thể hiện rõ so sánh kiến trúc chính của paper.

## Mapping sang notebook

Notebook được chia theo flow đơn giản bạn yêu cầu:

- `Load`: config, imports, dataset contract, image/caption loader.
- `EDA`: thống kê caption length và kiểm tra ảnh mẫu.
- `Model`: BPE tokenizer, dataloader, ResNet+LSTM baseline, ImageHRM ResNet.
- `Eval`: training loop, validation loss, greedy decoding, ROUGE-L.
- `Test`: smoke test, held-out prediction examples, bảng metric cuối.

## Data flow hiện tại

Notebook đã được đổi sang production-first:

1. Mặc định hiện tại là `DATA_MODE = "subset_1gb"` để tải trước khoảng 1GB dữ liệu thật và test train.
2. Subset được ghi vào `data/rocov2_subset_1gb`, không trộn với full data.
3. Subset mặc định tải 1 shard train và 1 shard test, khoảng 0.93GB parquet. Để giữ dưới khoảng 1GB, `val.csv` tạm reuse từ `test.csv`.
4. Nếu muốn full production data, đổi `DATA_MODE = "local"` hoặc `DATA_MODE = "download_full"`.
5. Với `local`, notebook ưu tiên dùng `data/rocov2/train.csv`, `data/rocov2/val.csv`, `data/rocov2/test.csv`. Nếu chưa có và `AUTO_DOWNLOAD_FULL_DATA = True` thì notebook tự tải full dataset `eltorio/ROCOv2-radiology`.

Hai mode còn lại vẫn giữ để debug có chủ đích:

1. `hf_sample`: tải sample nhỏ từ `eltorio/ROCOv2-radiology` về `_hf_sample_roco`.
2. `smoke`: tạo `_smoke_roco` để test wiring.

Training cũng đã đổi sang production-default:

1. `subset_1gb` dùng `SUBSET_EPOCHS = 3` để test train trước.
2. Full/local data dùng `EPOCHS_TO_RUN = NUM_EPOCHS = 50`.
3. Không còn tự hạ xuống 1 epoch chỉ vì đang dùng sample mode.

Notebook cũng đã sửa root path để không phụ thuộc nơi bấm `Run All`. Nó dò ngược lên project root dựa trên `pyproject.toml` và folder `RBL1/imagehrm_resnet_reproduce`.

## Các điểm chưa thể tái lập tuyệt đối

Các điểm này chưa giải quyết được vì paper không công bố đủ chi tiết:

- Nguồn download và layout ROCOv2 chính xác của tác giả.
- Batch size.
- Learning rate và weight decay.
- BPE vocab size/training recipe.
- Công thức ACT loss chính xác.
- Package/config CIDEr chính xác.
- Generation settings ngoài sequence length 512.
- Paper có mismatch nội bộ: implementation details nói H=2/M=2/L=2 layers, nhưng bảng kết quả ResNet triple-loop ghi H/M/L config 1/1/1.

Notebook vì vậy để các lựa chọn còn thiếu trong config cell. Muốn claim full metric reproduction thì cần đối chiếu với code hoặc experiment log gốc của tác giả.
