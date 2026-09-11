# Delivery: GA → GNN → guided GA (C++20)

Baseline phân bổ **GoodsSlot → ReceivePoint**, triển khai theo `draft_for_gpt_astra.pdf` và sơ đồ được cung cấp. Repo ban đầu chỉ có README, nên GA dưới đây là **GA baseline mới**, chưa phải bản tái hiện một GA production có sẵn.

- GA tạo nhãn chỉ khi đáp ứng chính xác nhu cầu, đúng type/range/angle và **0 giao cắt**.
- GNN LibTorch học phân phối theo từng slot; có lớp `UNUSED` khi dư hàng.
- GNN chỉ cung cấp prior cho khởi tạo/mutation. GA sửa nghiệm, đánh giá giao cắt và kiểm tra kết quả cuối.
- Toàn bộ domain, graph, training, inference, serialization, generation và evaluation dùng C++. Không dùng Python.

## Build

Cần compiler hỗ trợ C++20, CMake >=3.20. Phần neural cần bản **LibTorch CPU C++11 ABI**, được cấu hình theo [hướng dẫn chính thức PyTorch](https://docs.pytorch.org/cppdocs/installing.html). CI ghim bản 2.7.1; có thể chọn bản phù hợp hệ điều hành của bạn từ trang chính thức.

Trên Fedora, cài công cụ build:

```bash
sudo dnf install gcc-c++ cmake make unzip
```

Chỉ GA/domain, không cần LibTorch:

```bash
cmake -S . -B build-core -DDELIVERY_WITH_TORCH=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-core -j2
ctest --test-dir build-core --output-on-failure
```

Đủ pipeline, sau khi giải nén LibTorch:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build build -j1
ctest --test-dir build --output-on-failure
```

Dùng `-j1` để giảm RAM lúc biên dịch LibTorch headers. `DELIVERY_WITH_TORCH=ON` là mặc định và báo lỗi rõ nếu thiếu dependency, không thay thế GNN bằng một heuristic.

## Chạy pipeline

Từ thư mục gốc repo:

```bash
# Split toàn bộ instance trước khi chạy GA: train 60%, val 20%, test 10%, OOD 10%.
build/generate_dataset data/generated/run1 100 1000 configs/baseline.cfg

# TRAIN_DIR VAL_DIR MODEL EPOCHS SEED HIDDEN LAYERS BATCH_SIZE LR
build/train_gnn data/generated/run1/train data/generated/run1/val models/baseline.pt 50 42 64 2 8 0.001

# So sánh GA, GNN-GA init, GNN-GA init+mutation trên cùng instance/seed/caps.
build/evaluate_gnn_ga data/generated/run1/test models/baseline.pt configs/baseline.cfg 42 10 results/matched-test.csv
build/evaluate_gnn_ga data/generated/run1/ood models/baseline.pt configs/baseline.cfg 42 10 results/matched-ood.csv
```

GA-only:

```bash
build-core/generate_dataset data/generated/core1 30 1000 configs/baseline.cfg
build-core/evaluate_ga data/generated/core1/test configs/baseline.cfg 42 3 results/ga-test.csv
```

Generator ghi cả instance không tìm được nhãn, với label rỗng. Training chỉ đọc record đã chứng nhận lại; evaluation đọc **tất cả** instance trong split, tránh loại các bài khó khỏi tập test. Mỗi lần chạy generator dùng thư mục mới để tránh dữ liệu cũ lẫn vào split. Training từ chối ID trùng giữa train/val. Khi nhập dữ liệu thực, người dùng phải giữ ID ổn định và loại bản sao cùng instance trước khi chia tập.

## API tích hợp

```cpp
#include "delivery/gnn.hpp"
#include "delivery/io.hpp"

// Populate trucks, slots and receivers from your application's data.
delivery::Instance instance = delivery::read_record("instance.delivery").instance;
auto model = delivery::load_model("models/baseline.pt");
auto prior = model->predict(instance); // [slot][receiver + UNUSED]
auto result = delivery::run_ga(instance, delivery::GAConfig{}, 42, &prior);
if (result.validation.feasible) {
    delivery::write_allocation("allocation.csv", instance, result.allocation);
}
// result.validation.certified() additionally requires zero crossings.
```

Các project khác có thể `add_subdirectory(delivery)` rồi link `delivery_core` hoặc `delivery_gnn`. Cần cùng LibTorch ABI cho tất cả module C++.

| Module | Vai trò |
|---|---|
| `include/delivery/domain.hpp`, `src/domain.cpp` | Struct, eligibility duy nhất, validator, hình học, graph/features |
| `include/delivery/ga.hpp`, `src/ga.cpp` | GA, prior sampling, matching repair, ngân sách và thống kê |
| `include/delivery/io.hpp`, `src/io.cpp` | Định dạng dữ liệu, config, generator, CSV allocation |
| `include/delivery/gnn.hpp`, `src/gnn.cpp` | LibTorch MPNN hai chiều, masked edge scores, checkpoint |
| `apps/` | Dataset generation, training, GA evaluation, matched GNN-GA evaluation |
| `tests/` | Domain/GA tests và neural integration tests |

## Quy tắc domain và hình học

`Allocation[s] = r` hoặc `-1` (UNUSED), nên không thể gán một slot nhiều lần. Ma trận trong sơ đồ là `X[r][s] = (Allocation[s] == r)`; prior trong API lưu theo hàng slot để thuận tiện softmax. `sum_r X[r][s] <= 1`, không yêu cầu dùng hết hàng dư. Mỗi loại hàng tại mỗi receiver phải được giao **đúng** nhu cầu, không giao thừa.

Range tính từ vị trí truck sở hữu slot. Góc dùng radian, `heading` đo từ trục +x ngược chiều kim đồng hồ, `half_angle` là nửa góc mở; pi nghĩa là không hạn chế hướng. `eligible()` là nguồn quy tắc duy nhất cho graph và GA. Type hiện là vocabulary cố định `0..K-1`; số truck/slot/receiver biến thiên. Checkpoint yêu cầu giữ nguyên ý nghĩa type và K.

Mỗi cặp `(truck, receiver)` chỉ tạo một đoạn thẳng, dù nhiều slot đi cùng tuyến. Chỉ đếm giao cắt **proper interior**. Shared endpoint, tiếp xúc đầu mút với nội đoạn, trùng tuyến và chồng lấn thẳng hàng không bị tính. Đây là lựa chọn hình học của baseline; nếu production coi collinear overlap là vi phạm, cần sửa `interior_crossing()` và test trước khi tạo lại nhãn. Range/angle dùng tolerance 1e-9; orientation dùng long double, không phải exact arithmetic cho tọa độ cực đoan.

GA có tournament selection, uniform crossover, mutation, elitism, deduplication và augmenting-path matching repair theo demand token. Repair ưu tiên lựa chọn hiện tại, có thể thay đổi đề xuất của GNN để khôi phục tính khả thi. Guided/random mixture có thành phần uniform bắt buộc; crossover không được GNN định hướng, theo baseline tối thiểu trong PDF. Không thêm objective khoảng cách ngoài crossing count.

## Neural model

Slot features: tọa độ, min/max range, sin/cos heading, half-angle, one-hot goods type. Receiver features: tọa độ và nhu cầu theo type. Edge features: dx, dy, distance, distance/max-range, sin/cos relative bearing. Tọa độ/range chuẩn hóa theo scale của instance, demand chia số slot; không dùng database ID hoặc index làm feature.

MLP encoders → L lớp message passing SR/RS với mean aggregation → residual MLP + LayerNorm → MLP edge scorer và UNUSED scorer. Logits cạnh không hợp lệ là `-infinity`; softmax theo **các receiver hợp lệ + UNUSED cho mỗi slot**. Không có cạnh vẫn có UNUSED xác suất 1. Loss là categorical cross-entropy trung bình theo slot, rồi trung bình theo graph trong batch. Training chạy CPU, Adam, shuffle cố định seed, lưu model tốt nhất theo validation NLL. `.pt.meta` lưu kích thước model; `.pt.csv` lưu train/val NLL, top-1 và top-3. Giữ `.pt` và `.pt.meta` cùng nhau. Chưa hỗ trợ resume optimizer, GPU hoặc batching graph bằng sparse tensor.

## Dữ liệu và protocol

`.delivery` là định dạng text versioned, đọc/ghi bằng C++, whitespace-separated, số thực lưu 17 chữ số:

1. `DELIVERY_V1`
2. `"instance-id" K T S R`
3. T dòng: `x y min_range max_range heading half_angle`
4. S dòng: `truck_index type_index`
5. R dòng: `x y demand[0] ... demand[K-1]`
6. `label_seed label_budget label_seconds label_length`
7. `label_length` số nguyên destination; `0` length nghĩa là chưa có nhãn, S nghĩa là nhãn chứng nhận.

Ví dụ nhập nằm ở `configs/example.delivery`. `manifest.csv` ghi kích thước, số cạnh, nhu cầu, seed, budget, certification và thời gian. Generator dùng hai dải vị trí với jitter, cung dư và các size train 3–5 truck/3–6 receiver; OOD 8 truck/10 receiver. Đây là synthetic smoke benchmark, chưa đại diện dữ liệu thực.

Mỗi phương pháp dùng cùng **cap số fitness evaluation duy nhất** và cap số proposal. Dừng sớm khi tìm thấy zero-crossing; số evaluations thực tế do đó có thể khác nhau. Duplicate allocation không tính thêm fitness evaluation, vẫn tính proposal. Nếu không gian nhỏ hoặc proposal cap hết, số evaluation có thể thấp hơn budget: đọc cả hai cột, không hiểu là đã dùng hết budget. `first_zero=-1` là chưa tìm thấy. CSV báo repair/failure counts để tính tỷ lệ theo proposals. Thời gian triển khai của GNN-GA cộng inference + search; training là chi phí offline riêng. Reproducibility bảo đảm trong cùng toolchain/library, không cam kết bitwise giữa mọi hệ thống.

**Nhãn GA là một nghiệm tối ưu cho riêng objective giao cắt khi đạt 0**, không phải ground truth duy nhất. Positive crossing không được chứng nhận. Không kết luận GNN-GA tốt hơn GA chỉ từ accuracy hoặc một seed. Cần xem success rate, first-zero evaluations, best crossings, runtime và phân phối trên matched seeds, phân tầng theo size.

## Trạng thái kiểm chứng

Xem `results/VALIDATION.md` cho những gì thực sự đã chạy. GitHub Actions có hai job: core độc lập và full LibTorch (unit tests + generate/train/evaluate smoke). Khi dùng một commit, kiểm tra kết quả Actions của đúng commit đó. Chưa có kết quả nào chứng minh lợi ích GNN-GA; CSV GA smoke không được xem là nghiên cứu so sánh.

## Giải một instance và xuất nghiệm cuối

```bash
# Không cần model
build-core/solve_delivery configs/example.delivery configs/baseline.cfg 42 results/allocation.csv

# Có model: thêm file allocation.csv.prior.csv để xem xác suất (receiver=-1 là UNUSED).
build/solve_delivery data/generated/run1/test/synthetic-1080.delivery configs/baseline.cfg 42 results/allocation.csv models/baseline.pt
```

`solve_delivery` trả exit code 0 khi certified; 3 khi có nghiệm khả thi nhưng vẫn có giao cắt (CSV vẫn được ghi); 1 khi không có nghiệm khả thi hoặc lỗi input; 2 khi sai cú pháp. Tên instance trong ví dụ thứ hai ứng với generator COUNT=100, SEED=1000 ở trên.
