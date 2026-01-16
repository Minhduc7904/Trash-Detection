# BÁO CÁO TỰ ĐỘNG GÁN NHÃN DỮ LIỆU SỬ DỤNG YOLOV8

## 1. Mục tiêu

### 1.1. Mục tiêu chính
- Thực hiện tự động gán nhãn dữ liệu nhằm giảm thời gian và công sức gán nhãn thủ công
- Tạo tập dữ liệu có nhãn chất lượng cao để phục vụ huấn luyện và cải thiện mô hình phân loại rác thải

### 1.2. Mục tiêu cụ thể
- Gán nhãn tự động cho các đối tượng: **GLASS** (thủy tinh), **PAPER** (giấy), **PLASTIC** (nhựa)
- Đảm bảo ngưỡng tin cậy (confidence threshold) ≥ 0.6 để giảm thiểu false positives
- Tạo dataset theo định dạng YOLO để dễ dàng sử dụng cho việc huấn luyện tiếp theo

---

## 2. Cách tiếp cận

### 2.1. Mô hình sử dụng
- **Mô hình**: YOLOv8s (Small variant)
- **Trọng số**: `refine_last_phase3_10epochs.pt`
- **Ngưỡng confidence**: 0.6
- **Thiết bị**: CPU

### 2.2. Quy trình tự động gán nhãn

```
┌──────────────────────┐
│  Unlabeled Images    │
│   (Ảnh chưa nhãn)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   YOLOv8 Inference   │
│  (conf ≥ 0.6)       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Filter by classes   │
│ GLASS/PAPER/PLASTIC  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Save to dataset/    │
│  - images/train/     │
│  - labels/train/     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Delete processed    │
│  from unlabeled/     │
└──────────────────────┘
```

### 2.3. Quy trình kiểm tra chất lượng
1. **Tự động gán nhãn** bằng script `trash_detection.py`
2. **Kiểm tra thủ công** sử dụng công cụ LabelImg
3. **Chỉnh sửa và cải thiện** các nhãn chưa chính xác
4. **Thống kê lỗi** để cải thiện mô hình trong lần huấn luyện tiếp theo

---

## 3. Tiến độ hiện tại

### 3.1. Tổng quan xử lý

| Chỉ số | Số lượng | Ghi chú |
|--------|----------|---------|
| **Ảnh đã xử lý tự động** | 158 | Đã tạo file ảnh và nhãn |
| **Ảnh còn lại chưa xử lý** | 12 | Trong thư mục `unlabeled_images/` |
| **File nhãn đã tạo** | 158 | Format YOLO (.txt) |
| **Tỷ lệ hoàn thành** | 92.94% | 158/(158+12) × 100% |

### 3.2. Chi tiết ảnh còn lại chưa xử lý

Các ảnh sau **không** được mô hình detect hoặc không có đối tượng thuộc 3 class target:

```
1. 1b128cc5c47e.jpg
2. 429d80c54519.jpg
3. 47cff746358f.jpg
4. 6fd1671b8f5e.jpg
5. 96b8e2260d08.jpg
6. d89718cf1e6a.jpg
7. f05f7019e010.jpg
8. f85872b46f22.jpg
9. fc06d12f3acd.jpg
10. nhua difficult copy.jpg
11. nhua difficult.jpg
12. plastic3.jpg
```

**Lý do có thể:**
- Không có đối tượng GLASS/PAPER/PLASTIC trong ảnh
- Đối tượng quá nhỏ hoặc bị che khuất
- Góc chụp khó, ánh sáng kém
- Confidence score < 0.6

### 3.3. Cấu trúc thư mục output

```
dataset/
├── images/
│   └── train/          # 158 ảnh .jpg
└── labels/
    └── train/          # 158 file .txt + 1 classes.txt
```

---

## 4. Kết quả bước đầu

### 4.1. Thống kê sau kiểm tra thủ công bằng LabelImg

Sau khi review lại 158 ảnh đã được gán nhãn tự động, phát hiện:

| Loại lỗi | Số lượng | Tỷ lệ lỗi | Mô tả |
|----------|----------|-----------|-------|
| **Thiếu object** | 18 | 11.39% | Mô hình bỏ sót đối tượng trong ảnh |
| **Sai object** | 6 | 3.80% | Phân loại sai class (ví dụ: PAPER → PLASTIC) |
| **Bounding box sai** | 5 | 3.16% | Vị trí hoặc kích thước box không chính xác |
| **Tổng lỗi** | 29 | 18.35% | Trên 158 ảnh đã xử lý |

### 4.2. Đánh giá độ chính xác

```
✅ Ảnh hoàn toàn chính xác: ~129/158 (81.65%)
⚠️  Ảnh cần chỉnh sửa: 29/158 (18.35%)
```

### 4.3. Phân tích nguyên nhân lỗi

#### 4.3.1. Thiếu object (18 trường hợp)
- Đối tượng bị che khuất một phần
- Đối tượng quá nhỏ hoặc ở góc ảnh
- Ánh sáng kém, độ tương phản thấp
- Mô hình chưa được huấn luyện đủ với các trường hợp này

#### 4.3.2. Sai object (6 trường hợp)
- Nhầm lẫn giữa PAPER và PLASTIC (vật liệu tương đồng)
- Đối tượng có đặc điểm hỗn hợp (bao bì có nhiều lớp vật liệu)
- Ánh sáng ảnh hưởng đến màu sắc và kết cấu

#### 4.3.3. Bounding box sai (5 trường hợp)
- Box quá rộng/hẹp
- Không bao trọn đối tượng
- Bao cả background không cần thiết

---

## 5. Hạn chế

### 5.1. Hạn chế của phương pháp tự động

| Hạn chế | Ảnh hưởng | Giải pháp |
|---------|-----------|-----------|
| **Confidence threshold cố định** | Bỏ sót đối tượng có độ tin cậy thấp | Thử nghiệm với ngưỡng thấp hơn (0.4-0.5) |
| **Không xử lý được ảnh khó** | 12 ảnh không được gán nhãn | Gán nhãn thủ công cho các ảnh này |
| **Lỗi phân loại** | 3.8% ảnh bị sai class | Huấn luyện thêm với hard examples |
| **Bounding box không chính xác** | 3.16% cần điều chỉnh | Fine-tune IoU threshold |

### 5.2. Hạn chế của mô hình

1. **Độ chính xác chưa cao hoàn toàn**
   - 18.35% ảnh cần chỉnh sửa sau khi auto-label
   - Cần human-in-the-loop để đảm bảo chất lượng

2. **Khả năng tổng quát hóa**
   - Gặp khó khăn với các trường hợp đặc biệt (ánh sáng kém, góc chụp lạ)
   - Cần mở rộng dữ liệu huấn luyện đa dạng hơn

3. **Class imbalance**
   - Không rõ phân bố số lượng mỗi class trong 158 ảnh
   - Có thể thiên lệch về một class nào đó

### 5.3. Hạn chế về quy trình

1. **Thiếu validation set**
   - Chỉ xuất ra `train/`, chưa có `val/` và `test/`
   - Khó đánh giá hiệu năng trên tập mới

2. **Không track metrics**
   - Chưa có log về precision, recall, mAP
   - Khó so sánh giữa các phiên bản mô hình

3. **Xử lý ảnh lỗi chưa tối ưu**
   - Ảnh không detect được chỉ in warning
   - Nên lưu vào thư mục riêng để xử lý sau

---

## 6. Đề xuất cải thiện

### 6.1. Ngắn hạn
- [ ] Gán nhãn thủ công cho 12 ảnh còn lại
- [ ] Chỉnh sửa 29 ảnh có lỗi đã phát hiện
- [ ] Chia dataset thành train/val/test (70/20/10)

### 6.2. Trung hạn
- [ ] Thử nghiệm với ngưỡng confidence thấp hơn (0.4-0.5)
- [ ] Sử dụng các kỹ thuật augmentation để tăng độ đa dạng
- [ ] Thêm active learning: chọn các ảnh khó để gán nhãn thủ công và huấn luyện lại

### 6.3. Dài hạn
- [ ] Huấn luyện lại mô hình với dữ liệu đã chỉnh sửa
- [ ] Đánh giá mô hình trên validation set và test set
- [ ] Xây dựng pipeline tự động: auto-label → review → retrain → evaluate

---

## 7. Kết luận

Quá trình tự động gán nhãn bằng YOLOv8 đã đạt được:

✅ **Thành công**:
- Xử lý tự động được **158/170 ảnh** (92.94%)
- Tiết kiệm đáng kể thời gian so với gán nhãn hoàn toàn thủ công
- Tạo được baseline dataset cho việc huấn luyện tiếp theo

⚠️ **Cần cải thiện**:
- 18.35% ảnh cần chỉnh sửa → Chưa đạt chất lượng production
- 12 ảnh không xử lý được → Cần gán nhãn thủ công
- Cần tích hợp validation và evaluation vào quy trình

🎯 **Hướng đi tiếp theo**:
Kết hợp giữa **tự động hóa** (tiết kiệm thời gian) và **kiểm tra thủ công** (đảm bảo chất lượng) là phương pháp tối ưu để xây dựng dataset chất lượng cao phục vụ cho mô hình phân loại rác thải.

---

**Ngày tạo báo cáo**: 12/01/2026  
**Người thực hiện**: Auto-labeling với YOLOv8s  
**Công cụ review**: LabelImg  
**Mô hình**: refine_last_phase3_10epochs.pt
