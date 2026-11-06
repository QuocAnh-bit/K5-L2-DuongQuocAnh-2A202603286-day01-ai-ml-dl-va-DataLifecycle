# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
{
    "taxonomy_name": "ImageNet-1K",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
}
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Cả bức ảnh, vì nếu là một vật thể riêng ở trong ảnh record phải có tọa độ riêng hoặc kích thước riêng
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- đi kèm taxomy/checkpoint 
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- vì mỗi class đảm đương 1 nhiệm vụ khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Phải quy định rõ ràng quy tắc lớp đại diện cho toàn bộ bức ảnh
6. Vì sao không được lấy `score` của model làm ground truth?
- Bởi vì đó là độ tự tin của model đối với dự đoán, không chứng minh được điều đó đúng

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
{
    "class_name": "person",
    "score": 0.610931,
    "bbox_xyxy": [
      0.12,
      263.18,
      61.55,
      310.78
    ],
    "bbox_width": 61.44,
    "bbox_height": 47.59
  }

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên-trái: (0.12, 263.18);
- Góc dưới-phải: (61.55, 310.78);
- Rộng: 61.44px
- Cao: 47.59px

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- 0.2: 11 prediction
- 0.6: 6 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng giúp giảm nguy cơ bỏ sót nhưng làm tăng khối lượng kiểm duyệt lên

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Bounding box phải bao sát toàn bộ phần vật thể nhìn thấy trong ảnh, không bao gồm quá nhiều nền xung quanh hoặc các vật thể khác. Tọa độ box phải chính xác theo thứ tự [x1, y1, x2, y2], trong đó (x1, y1) là góc trên bên trái và (x2, y2) là góc dưới bên phải.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Cái nồi ở góc dưới bàn gỗ
- Bởi vì không biết có cần gán nhãn vật thể chỉ nhìn thấy một phần hay không và mỗi box cần bao nhiêu phần nhìn thấy
để ước lượng toàn bộ vật thể

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-007; `class_name`=oven ; `score`=0.489667 ; số điểm polygon=`polygon_point_count`=213 ; 3–5 điểm đầu của `polygon_xy`= [[1.0, 189.0],[1.0, 118.0],[1.0,214.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Bounding box: Chỉ xác định vùng hình chữ nhật bao quanh vật thể.
- Polygon: Mô tả đường biên thực tế của vật thể bằng nhiều điểm tọa độ.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- mã định danh riêng cho từng đối tượng cụ thể trong dữ liệu

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Người gán nhãn phải vẽ polygon bám sát đường viền thực tế của vật thể, chỉ bao gồm phần vật thể nhìn thấy được. Không đưa phần nền hoặc vật thể bên cạnh vào mask. Các điểm polygon phải liên tục, không tự ý tạo khoảng trống hoặc đường biên cắt ngang vật thể.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Cái bàn và các vật dụng có trên bàn
- escalation lên lab coach nếu guideline chưa quy định rõ

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth | Điểm cần chú ý | Annotator | Reviewer |
|---|---|---|---|---|
| Phân loại ảnh | Một ảnh → một nhãn lớp. | `traffic`: `cab`, score `0.510915`; ảnh có thể có nhiều chủ thể. | Chọn lớp chính theo guideline. | Kiểm tra nhãn đúng nội dung ảnh. |
| Phát hiện vật thể | Mỗi vật thể → class + bounding box. | `kitchen`: `person`; box nằm sát mép trái ảnh. | Gán class và vẽ box sát vật thể. | Kiểm tra class, vật thể bị sót và độ chính xác của box. |
| Instance segmentation | Mỗi vật thể → class + instance ID + mask. | `kitchen`: `oven`, `kitchen-007`, polygon có 213 điểm. | Vẽ mask riêng, theo đường biên nhìn thấy. | Kiểm tra mask có đúng biên và không lẫn vật thể khác. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ dùng dữ liệu cho mục đích bài lab; không tự ý sao chép hoặc chia sẻ.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Gặp dữ liệu ngoài phạm vi: dừng xử lý và báo Lab Coach hoặc người phụ trách.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
