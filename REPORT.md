Họ tên: Hồ Đăng Phúc
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – | – | 0.084 | 0.060–0.113 | 0.90 | 0.132 | 0.028 | 0.172 |
| E1 | LoRA LLM | 33,030,144 | 6.91 | 44.6 | 0.269 | 0.231–0.308 | 1.19 | 0.157 | 0.208 | 0.514 |
| E2 | + projector | 142,135,296 | 8.73 | 45.2 | 0.337 | 0.297–0.382 | 1.04 | 0.164 | 0.283 | 0.603 |
| E3 | + LoRA ViT | 148,426,752 | 8.92 | 49.8 | **0.409** | 0.366–0.453 | 0.88 | 0.279 | 0.326 | 0.681 |
| E4 | `data_size` = 0.1 (setup E3) | 148,426,752 | 8.33 | 49.6 | 0.281 | 0.247–0.318 | 1.64 | 0.177 | 0.205 | 0.569 |
| E4 | `data_size` = 1.0 (= E3) | 148,426,752 | 8.92 | 49.8 | 0.409 | 0.366–0.453 | 0.88 | 0.279 | 0.326 | 0.681 |

Phần cứng: Kaggle Tesla T4 16 GB, QLoRA NF4 (LLM 4-bit, ViT + projector giữ fp16/fp32), 300 bước, fp16 autocast + GradScaler.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Đếm trực tiếp trên model (tổng 4826.8M):
- **ViT** (`model.visual` trừ merger): ~306.2M. 
- **Projector**: 109.1M, gồm `merger` 27.3M và 3 `deepstack_merger` 81.8M (Qwen3-VL đưa đặc trưng ViT tầng giữa vào nhiều tầng LLM).
- **LLM**: 4022.5M, cộng `lm_head` 389.0M (không tied với embedding).

Mỗi setup cập nhật:
- **E1**: chỉ LoRA trên các Linear của LLM, 33.0M tham số. ViT, projector và LLM gốc đều đóng băng.
- **E2**: thêm toàn bộ projector, 33.0M + 109.1M = 142.1M.
- **E3**: thêm LoRA trên các Linear của ViT, 142.1M + 6.3M = 148.4M (tổng cộng 348 module LoRA).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` = **0.961**, tức 12/311 ảnh có output hỏng. Hai lỗi chính:

1. **Lặp box crack vô hạn.** Model sinh chuỗi box crack trượt dần theo một đường chéo cho tới khi chạm `max_new_tokens=256`, nên JSON bị cắt giữa chừng. Đây là toàn bộ 12 ca hỏng. Ví dụ output thô:
   `[{"label":"crack","bbox_2d": [559, 375, 600, 407]}, {"label":"crack","bbox_2d": [575, 407, 610, 434]}, {"label":"crack","bbox_2d": [600, 434, 635, 460]}, … [705, 512, 740, ` (bị cắt).
2. **Bỏ sót nặng và box lệch.** Trung bình model chỉ trả ~1 box mỗi ảnh, nên trên 583 box GT chỉ có 38 TP (FN = 545). Riêng crack chỉ được 7 TP trên 299, F1 = 0.028: model khoanh crack thành box nhỏ, trong khi nhãn GT là box dài bao cả vết nứt.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Chấm lại cùng file dự đoán với `--coord pixel` cho **F1 = 0.000** ở cả E0 (bản norm1000 là 0.084) và E3 (bản norm1000 là 0.409).

Lý do: Qwen3-VL xuất toạ độ chuẩn hoá về 0–1000. Khi coi đó là pixel, `score.py` chia cho width/height (ví dụ 1920×1080) rồi nhân 1000, nên mọi box co về góc trên trái. IoU với GT gần như bằng 0, nghĩa là mọi dự đoán thành FP và mọi GT thành FN.

Cách kiểm chứng:
- Chạy `scripts/score.py --coord pixel` trên `work/e0.jsonl` và `work/e3.jsonl`.
- Trong notebook có round-trip check: chuyển YOLO sang 0–1000 rồi chuyển ngược về, so với box gốc đã kẹp vào khung ảnh.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Mục tiêu là học P(JSON | ảnh, prompt), nên chỉ các token trả lời mới mang tín hiệu cần học. Phần prompt và token ảnh được đặt label = −100.

Nếu tính loss trên cả những phần này:
- **Token ảnh chiếm phần lớn chuỗi** (hàng trăm token cho mỗi ảnh). Loss trên chúng vừa vô nghĩa, vì đó là placeholder `<|image_pad|>` không dự đoán được nội dung, vừa lấn át gradient của vài chục token JSON.
- **Prompt cố định nên rất dễ đoán.** Loss tụt nhanh và trông như model hội tụ, nhưng thực ra nó chỉ học thuộc prompt, còn chất lượng box không tăng.
- **Hệ quả**: loss thấp giả, F1 thấp hơn, và model có thể bắt đầu sinh lại prompt trong output.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Projector nằm trong `model.visual` (gồm `merger` và `deepstack_merger_list`). Nếu ViT bị lượng tử hoá, các Linear của projector cũng thành `Params4bit` (uint8), và tensor kiểu nguyên thì không thể `requires_grad`. Tôi đã gặp lỗi đúng như vậy: *"only Tensors of floating point … can require gradients"*. Lần chạy lỗi trước đó đếm được E2 chỉ có 87.6M tham số (= 33.0M + 109M/2) thay vì 142M, cho thấy projector đã bị nén 4-bit.

Nguyên nhân gốc: ở transformers 5.x, `llm_int8_skip_modules=['visual']` khớp theo prefix/suffix chứ không theo substring, nên không bỏ qua được `model.visual.blocks…`.

Cách sửa:
- Thêm `'model.visual'` vào danh sách skip.
- `assert` không còn `Linear4bit` nào có tên chứa `visual`.
- Tạo lại mỗi tham số trainable bằng `nn.Parameter(p.detach().float())`. Không đổi `p.data` tại chỗ, vì làm vậy `grad_dtype` vẫn là Half và gây lỗi *"assign float grad to grad_dtype Half"*.

Projector phải ở fp32 vì GradScaler từ chối unscale gradient fp16 (*"Attempting to unscale FP16 gradients"*). Ngoài ra, cập nhật Adam trên trọng số fp16 bị mất bước nhỏ do underflow hoặc làm tròn. Forward vẫn chạy dưới autocast fp16, nên chi phí VRAM chỉ tăng 6.91 → 8.73 GB.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Ảnh được cắt từ video (`vlcsnap-…`). Các khung trong cùng một phiên quay gần như trùng nhau: cùng đoạn đường, cùng ổ gà, cùng ánh sáng. Nếu chia ngẫu nhiên, ảnh test gần như đã xuất hiện trong train, model chỉ cần nhớ là đủ, và F1 đo được thực chất là độ ghi nhớ chứ không phải khả năng tổng quát sang đoạn đường mới. Chia theo phiên (`session_key`) buộc tập test là những con đường model chưa từng thấy.

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1422 | 1074 | 1750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
Có.

- **E1 → E2:** F1 tăng từ 0.269 lên 0.337 (+0.068). Đổi lại, VRAM tăng 1.8 GB (6.91 → 8.73) và thời gian gần như không đổi (+0.6 phút).
- **E2 → E3:** F1 tăng thêm lên 0.409 (+0.072), mean IoU tăng 0.680 → 0.708, và FP/ảnh giảm 1.04 → 0.88. Chi phí chỉ là +0.2 GB VRAM và +4.6 phút.

Khoảng tin cậy:
- E1 (0.231–0.308) và E3 (0.366–0.453) **không chồng lấn**, nên lợi ích của E3 so với E1 là chắc chắn.
- Từng bước liền kề thì CI chồng lấn một phần: E1/E2 chồng ở 0.297–0.308, E2/E3 chồng ở 0.366–0.382. Vì vậy riêng mỗi bước chỉ là bằng chứng vừa phải.

Lợi ích lớn nhất rơi vào pothole (0.164 → 0.279 ở bước E2 → E3). Cách giải thích hợp lý: ViT gốc chưa thấy kết cấu mặt đường hư hỏng, nên chỉnh ViT giúp ích nhiều hơn chỉ chỉnh LLM. Trên T4, mọi setup đều vừa 16 GB.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Bootstrap theo ảnh (1000 lần) cho CI95 rộng khoảng **±0.04**: E1 rộng 0.077, E2 rộng 0.085, E3 rộng 0.087.

Hệ quả cho việc so sánh:
- Chênh lệch dưới ~0.05 giữa hai setup không phân biệt được.
- Ngay các bước +0.07 cũng chỉ vừa đủ: CI của chúng vẫn chồng một phần.
- Kết luận chắc chắn chỉ có E0 < E1 < E3 và E4(0.1) < E3.

Để so sánh chính xác hơn, cần paired bootstrap trên cùng các mẫu resample (tính trực tiếp chênh lệch F1), chạy nhiều seed, hoặc một tập test lớn hơn. Ở đây mỗi setup chỉ chạy 1 seed, nên phương sai do huấn luyện chưa được tính vào CI.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E3, lớp kém nhất là **pothole (F1 0.279; TP 30 / FN 91 / FP 64)**, sau đó là crack (0.326). Manhole tốt nhất (0.681).

Pothole kém dù có nhiều mẫu train nhất (1074 box), vì:
- **Ngữ nghĩa mơ hồ:** ổ gà nông, vũng nước, mảng vá và bóng đổ trông rất giống nhau, nên model vừa bỏ sót vừa báo nhầm.
- **Biên mờ và box nhỏ:** IoU ≥ 0.5 khó đạt với vật thể không có đường viền rõ.
- **Lệch phân phối:** val chỉ có 66 box pothole, trong khi test có 121 box ở những đoạn đường khác.

Crack khó vì là vật thể dài và mảnh. Cách đóng box (một box dài hay nhiều box ngắn) không thống nhất giữa các nhãn, nên IoU dễ dưới 0.5 dù vị trí đúng: FP 189, FN 204.

Manhole tròn, có viền rõ, ít biến thể, nên dễ nhất.

Ảnh minh hoạ: `report/figures/e3_examples.png`, cùng các ca đúng (`e3_ĐÚNG_*.jpg`) và sai (`e3_SAI_*.jpg`). Đường loss ở `report/figures/loss.png`.

![E3 examples](report/figures/e3_examples.png)

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Tôi đổi biến **`data_size`** (tỉ lệ tập train), giữ nguyên setup E3 và cùng 300 bước. Giá trị 1.0 dùng lại kết quả E3.

| | F1 | CI95 | FP/ảnh |
|---|---|---|---|
| `data_size` 0.1 (~142 ảnh) | 0.281 | 0.247–0.318 | 1.64 |
| `data_size` 1.0 | 0.409 | 0.366–0.453 | 0.88 |

Kết luận đáng tin ở mức định tính: hai CI **tách rời nhau**, nên nhiều dữ liệu hơn chắc chắn tốt hơn.

FP/ảnh gần gấp đôi khi dùng 0.1. Vì số bước giữ nguyên, model lặp lại ~142 ảnh nhiều lần và overfit, nên báo box quá tay. Đó cũng là lý do thời gian không giảm (49.6 so với 49.8 phút).

Giới hạn của thí nghiệm:
- Chỉ có 2 điểm dữ liệu và 1 seed, nên chưa vẽ được đường cong theo data_size.
- Số bước và số epoch thay đổi cùng lúc, nên không tách được ảnh hưởng của "ít dữ liệu" với "overfit".

E4(0.1) vẫn tương đương E1 với full data (0.269). Nghĩa là mở ViT và projector bù được khoảng 10 lần dữ liệu.

## Nhìn lại

- **Bug âm thầm mà công cụ chấm không bắt được:** `llm_int8_skip_modules=['visual']` không hề bỏ qua ViT ở transformers 5.x. ViT và projector bị nén 4-bit mà không có cảnh báo nào. Tôi chỉ phát hiện được nhờ đếm tham số trainable (87.6M ≠ 142M) và thêm `assert`. Nếu thiếu bước đếm này, có thể E2 và E3 sẽ được chấm trên một model khác với điều báo cáo mô tả.
- **Lỗi E0 không thể hiện hết trong F1:** E0 bị cắt JSON do lặp vô hạn (12 ảnh). `score.py` xử lý những ca này như "không dự đoán", nên F1 không phân biệt được "model không thấy gì" với "model sinh rác". Một giới hạn `max_new_tokens` hoặc `repetition_penalty` có thể đổi kết quả E0 đáng kể.
- **Vấn đề môi trường:** Kaggle cài sẵn torchao 0.10, xung đột với peft khi thêm LoRA vào `nn.Linear` của ViT (ImportError ở E3), và phải gỡ torchao mới chạy được.
- **Điều tôi chưa hiểu rõ:** chưa rõ vì sao crack ở val (470 box) nhiều gấp đôi test. Có thể quy ước gán nhãn crack khác nhau giữa các phiên quay. Đây là nhiễu nhãn mà F1 không tách riêng được.
