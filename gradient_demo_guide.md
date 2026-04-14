# Gradient Dynamics — Script nói live coding

> Mỗi đoạn giải thích được gắn với đoạn code cụ thể.  
> Format: **`[code]`** → nói gì khi tay đang trỏ vào đoạn đó.

---

## Cell 1 — Imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
```
Phần này không có gì đặc biệt, em chỉ import những thứ cần dùng.

```python
torch.manual_seed(42)
```
Dòng này để đảm bảo mỗi lần chạy lại notebook cho cùng một kết quả — tránh trường hợp random mỗi lần một khác rồi khó so sánh giữa các thí nghiệm.

---

## Cell 2 — `SimpleMLP`

```python
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=3,
                 num_hidden_layers=4, activation='relu', norm_type=None):
```
Đây là class MLP trung tâm của toàn bộ demo. Thay vì viết cứng một kiến trúc, em cho phép truyền `activation` và `norm_type` vào như tham số — vì em sẽ dùng chính class này cho cả 5 thí nghiệm, chỉ đổi config, không đổi code.

```python
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
```
Với mỗi lớp ẩn, bước đầu tiên luôn là một lớp Linear — đây là phép biến đổi tuyến tính cơ bản.

```python
            if norm_type == 'batchnorm':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == 'layernorm':
                layers.append(nn.LayerNorm(hidden_dim))
```
Norm được đặt ngay sau Linear, *trước* activation. Thứ tự này quan trọng — Norm cần chuẩn hóa phân phối trước khi hàm kích hoạt xử lý. Nếu đặt sau ReLU, BatchNorm sẽ nhận đầu vào đã bị cắt mất phần âm, phân phối lệch, hiệu quả giảm rõ rệt.

```python
            if activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
```
Activation đặt cuối cùng trong mỗi lớp. Tóm lại thứ tự là: **Linear → Norm → Activation** — đây là convention chuẩn.

```python
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
```
Sau vòng lặp, thêm lớp output không có activation — CrossEntropyLoss của PyTorch sẽ tự xử lý softmax bên trong. Toàn bộ list layers được wrap bằng `nn.Sequential` để PyTorch tự gọi từng layer theo thứ tự — gọn hơn nhiều so với viết thủ công trong `forward()`.

---

## Cell 3 — `initialize_weights`

```python
def initialize_weights(model, init_type='he'):
    for module in model.modules():
        if isinstance(module, nn.Linear):
```
Hàm duyệt qua tất cả module trong model. Chỉ áp dụng cho lớp `Linear` — các lớp Norm và Activation không cần khởi tạo thủ công.

```python
            if init_type == 'xavier':
                nn.init.xavier_uniform_(module.weight)
```
**Xavier** được thiết kế cho activation đối xứng như tanh và sigmoid. Nó scale trọng số theo `2 / (fan_in + fan_out)` — mục tiêu là giữ phương sai của output xấp xỉ bằng phương sai của input, cả chiều forward lẫn backward.

```python
            elif init_type == 'he':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
```
**He** ra đời vì ReLU phá vỡ giả định của Xavier — ReLU cắt toàn bộ phần âm về 0, tức là một nửa tín hiệu bị mất sau mỗi lớp. He bù lại bằng cách scale theo `2 / fan_in` thay vì lấy trung bình. Đây là lý do khi dùng ReLU thì luôn dùng He, không dùng Xavier.

```python
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
```
**Small init** với std=0.01 — không phải chiến lược tốt, em dùng có chủ đích để tái hiện vanishing gradient ở thí nghiệm A. Trọng số quá nhỏ → activation đầu vào mỗi lớp rất nhỏ → gradient nhân qua 5 lớp co dần về 0.

```python
            nn.init.zeros_(module.bias)
```
Bias luôn khởi tạo bằng 0. Bias chỉ dịch chuyển đầu ra, không ảnh hưởng đến phân phối như weight, nên khởi tạo bằng 0 là convention phổ biến và không gây vấn đề gì.

---

## Cell 4 — `create_dummy_data` và `collect_gradient_norms`

```python
def create_dummy_data(num_samples=512, input_dim=20, num_classes=3):
    X = torch.randn(num_samples, input_dim)
    true_W = torch.randn(input_dim, num_classes)
    y = (X @ true_W).argmax(dim=1)
```
512 sample, mỗi sample 20 chiều. Nhãn được tạo bằng cách nhân X với một ma trận ngẫu nhiên rồi lấy argmax — tạo ra một "ground truth" tuyến tính ẩn bên trong để model học, không phải nhãn random vô nghĩa.

```python
def collect_gradient_norms(model):
    return {
        name: param.grad.norm().item()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
```
Đây là công cụ chẩn đoán chính. Sau khi `loss.backward()` chạy xong, PyTorch lưu gradient vào `param.grad` của từng tham số. Hàm này đọc ra L2 norm — tức `sqrt(Σ gᵢ²)` — của mỗi tensor và trả về dictionary. Em dùng norm vì nó cho một con số duy nhất đại diện cho độ lớn của cả tensor, và có thể so sánh giữa các lớp có kích thước khác nhau. Norm quá nhỏ → vanishing, quá lớn → exploding.

---

## Cell 5 — `train`

```python
def train(activation='relu', norm_type=None, init_type='he',
          lr=1e-3, gradient_clipping=False, num_epochs=5):
    X, y = create_dummy_data()
    model = SimpleMLP(num_hidden_layers=5, activation=activation, norm_type=norm_type)
    initialize_weights(model, init_type=init_type)
```
Khởi tạo dữ liệu, model, và áp dụng weight initialization theo config được truyền vào.

```python
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
```
CrossEntropyLoss cho bài toán phân loại, Adam là optimizer. `lr` được truyền từ ngoài vào để thí nghiệm E có thể dùng lr=0.1.

```python
        optimizer.zero_grad()
```
Phải gọi đầu mỗi vòng lặp — PyTorch cộng dồn gradient theo mặc định, không tự xóa. Nếu bỏ dòng này, gradient epoch sau cộng thêm vào epoch trước, toàn bộ quá trình học bị sai.

```python
        loss = criterion(model(X), y)
        loss.backward()
```
Forward pass rồi backward. Sau dòng `backward()` này, `param.grad` của tất cả parameter đã có giá trị — đây là lúc để đọc gradient.

```python
        grad_history.append(collect_gradient_norms(model))
        losses.append(loss.item())
```
Thu thập gradient norm ngay sau `backward()` và *trước* `optimizer.step()` — quan trọng vì sau khi `step()` cập nhật trọng số, gradient không còn phản ánh trạng thái trước khi cập nhật nữa.

```python
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Nếu bật clipping: tính tổng L2 norm của toàn bộ gradient trong model, nếu lớn hơn `max_norm=1.0` thì rescale tất cả xuống. Nó không thay đổi hướng gradient, chỉ giảm độ lớn — model vẫn học đúng hướng, chỉ bước đi nhỏ hơn.

```python
        optimizer.step()
```
Dùng gradient đã có — và đã được clip nếu bật — để cập nhật trọng số.

---

## Cell 6 — Chạy 5 thí nghiệm

```python
experiments = {
    'A: Sigmoid + small init':           dict(activation='sigmoid', norm_type=None,        init_type='small', lr=1e-3, gradient_clipping=False),
    'B: ReLU + He':                      dict(activation='relu',    norm_type=None,        init_type='he',    lr=1e-3, gradient_clipping=False),
    'C: ReLU + He + BatchNorm':          dict(activation='relu',    norm_type='batchnorm', init_type='he',    lr=1e-3, gradient_clipping=False),
    'D: ReLU + He + LayerNorm':          dict(activation='relu',    norm_type='layernorm', init_type='he',    lr=1e-3, gradient_clipping=False),
    'E: ReLU + He + Clipping (lr=0.1)':  dict(activation='relu',   norm_type=None,        init_type='he',    lr=0.1,  gradient_clipping=True),
}
```
Năm thí nghiệm, mỗi thí nghiệm là một dictionary config truyền thẳng vào hàm `train`.

**A** — Sigmoid + small init: được thiết kế để thất bại có chủ đích. Sigmoid bão hòa ở vùng giá trị lớn hoặc nhỏ, đạo hàm tiến về 0. Cộng thêm khởi tạo nhỏ, gradient nhân qua 5 lớp co dần về 0.

**B** — ReLU + He: baseline tốt nhất. Đây là điểm tham chiếu để so sánh với các thí nghiệm còn lại.

**C** — thêm BatchNorm: chuẩn hóa đầu ra theo batch statistics — mean và std tính trên toàn bộ batch. Gradient ổn định hơn, hội tụ nhanh hơn.

**D** — thay bằng LayerNorm: chuẩn hóa theo từng sample thay vì theo batch. Không phụ thuộc batch size — đây là lựa chọn mặc định trong Transformer.

**E** — lr=0.1 và clipping bật: learning rate gấp 100 lần bình thường. Không có clipping thì loss phân kỳ. Với clipping, model vẫn hội tụ nhưng không ổn định.

```python
results = {label: dict(zip(['losses', 'grad_history'], train(**cfg)))
           for label, cfg in experiments.items()}
```
Chạy tất cả bằng một dict comprehension, kết quả lưu vào `results` để dùng cho các biểu đồ.

---

## Cell 7 — Loss Chart

```python
ax.plot(range(1, len(data['losses']) + 1), data['losses'], ...)
```
Vẽ loss của từng thí nghiệm qua 5 epoch.

**Output thực tế:** A, B, C, D nằm sát nhau ở dưới, dao động quanh 1.0. Điều này không có nghĩa là B, C, D không học — mà 5 epoch quá ngắn để thấy sự khác biệt lớn. Bốn đường đó đều hoạt động đúng, chỉ là hội tụ nhanh từ epoch 1 rồi ổn định.

Điểm nổi bật nhất là thí nghiệm E — dao động cực mạnh: epoch 1 khoảng 1.5, epoch 2 vọt lên 22, epoch 3 rớt về 5, epoch 4 bùng lên 36, epoch 5 rớt xuống 11. Đây là biểu hiện điển hình của lr quá lớn — model liên tục nhảy qua lại quanh điểm tối ưu, không tìm được chỗ đứng. Clipping giữ nó không phân kỳ hoàn toàn, nhưng rõ ràng không đủ để ổn định — bằng chứng cho thấy clipping chỉ là lưới an toàn, không thay thế được việc chọn lr hợp lý.

---

## Cell 8 — Gradient Bar Chart

```python
    grads = data['grad_history'][0]   # epoch 1
    bars = ax.bar(range(len(grads)), list(grads.values()), ...)
    bars[0].set_edgecolor('black')    # viền đen cho lớp đầu
```
Vẽ gradient norm của từng tham số tại epoch 1. Cột đầu tiên viền đen là lớp đầu tiên — xa loss nhất, tín hiệu gradient phải đi qua nhiều lớp nhất mới tới đây.

**Output thực tế:**

**A (đỏ):** Toàn bộ cột gần bằng 0, trục Y chỉ đến 0.25. Các lớp ẩn giữa gần như không nhìn thấy, chỉ hai cột cuối gần output mới nhỉnh hơn một chút. Đây là vanishing gradient điển hình — tín hiệu chết dần từ output về lớp đầu.

**B (xanh lá):** Trục Y lên đến 2.0. Gradient tăng dần từ lớp đầu ra lớp gần output, dao động từ 0.25 đến gần 2.0. Tất cả các lớp đều nhận được tín hiệu.

**C (xanh dương) và D (tím):** Gradient lớp đầu ở mức 0.3–0.35, pattern giữa các lớp không đều như B — lúc tăng lúc giảm — do normalization thay đổi phân phối giữa các lớp.

**E (vàng):** Gradient lớn hơn hẳn so với B do lr=0.1 khiến loss lớn hơn, gradient theo đó cũng lớn hơn.

Nhìn A và B cạnh nhau là đủ thấy tất cả — cùng kiến trúc 5 lớp, chỉ đổi activation và init, gradient lớp đầu thay đổi từ gần 0 lên 0.25. Đó là lý do tại sao lựa chọn activation và initialization không phải chi tiết nhỏ.

---

## Cell 9 — Gradient Evolution Chart

```python
    norms = [g[list(g)[0]] for g in data['grad_history'] if g]
    ax.plot(range(1, len(norms) + 1), norms, ...)
```
Lấy gradient norm của tham số đầu tiên — tức lớp đầu tiên — tại mỗi epoch rồi vẽ đường theo thời gian.

**Output thực tế:**

**A, B, C, D:** Bốn đường nằm sát nhau dưới 0.4, gần như không thay đổi qua 5 epoch. A nằm sát đáy nhất — gradient lớp đầu gần bằng 0 xuyên suốt, lớp này không học gì từ đầu đến cuối.

**E:** Đường duy nhất nổi bật — epoch 1 khoảng 0.3, epoch 2 vọt lên 3.7, epoch 3 rớt về 1.0, epoch 4 bùng lên 5.2, epoch 5 rớt xuống 2.0. Pattern này ăn khớp hoàn toàn với biểu đồ loss ở Cell 7 — mỗi lần loss tăng vọt thì gradient lớp đầu cũng tăng vọt theo. Gradient phản ánh trực tiếp tình trạng training.

Kết hợp ba biểu đồ lại: Cell 7 cho biết model có học không, Cell 8 cho biết gradient phân phối như thế nào giữa các lớp tại epoch 1, Cell 9 cho biết gradient lớp đầu có ổn định không theo thời gian. Ba góc nhìn bổ sung cho nhau — xem một mình không đủ, phải xem cả ba mới thấy toàn cảnh.

---

## Bảng tra nhanh khi bị hỏi

| | Vanishing | Exploding |
|---|---|---|
| Nguyên nhân | Gradient nhân nhiều số < 1 | Gradient nhân nhiều số > 1 |
| Biểu hiện | Loss không giảm, gradient lớp đầu ≈ 0 | Loss tăng vọt hoặc NaN |
| Giải pháp | ReLU, He init, Normalization | Gradient clipping, lr nhỏ hơn |

| | Xavier | He |
|---|---|---|
| Scale | `2 / (fan_in + fan_out)` | `2 / fan_in` |
| Dùng với | tanh, sigmoid | ReLU, Leaky ReLU |

| | BatchNorm | LayerNorm |
|---|---|---|
| Chuẩn hóa theo | Batch | Từng sample |
| Mạnh khi | Batch lớn, CNN | Batch nhỏ, Transformer, RNN |
| Nhạy batch size | Có | Không |
