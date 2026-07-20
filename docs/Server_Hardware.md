# Thông số phần cứng server train

> Chỉ thông số máy. Hướng dẫn train 2 GPU: `SERVER_2x3090.md`. Gỡ nghẽn I/O: `BOTTLENECK_TRAIN_3090.md`.

---

## Tổng quan

| Hạng mục | Thông số |
|---|---|
| Vai trò | Training server (YOLO / deep learning) |
| GPU | **2 × NVIDIA GeForce RTX 3090** |
| VRAM (mỗi card) | **24 GB** (24576 MiB) |
| VRAM tổng | **48 GB** |
| OS | Ubuntu Linux, x86_64 |

---

## GPU

| Hạng mục | Thông số |
|---|---|
| Model | NVIDIA GeForce RTX 3090 |
| Số lượng | **2** |
| Index | GPU **0**, GPU **1** |
| Kiến trúc | Ampere (GA102) |
| VRAM / card | 24 GB GDDR6X |
| Power limit (điển hình) | ~350 W / card |
| CUDA | Hỗ trợ (driver máy chủ — kiểm tra `nvidia-smi`) |
| MIG | Không |

### Lệnh xác minh

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
```

---

## CPU

| Hạng mục | Thông số | Ghi chú |
|---|---|---|
| Model | *(điền từ `lscpu`)* | Trên host 2×3090 thường multi-core |
| Logical CPUs | *(vd. ~28 threads nếu htop hiện 0–27)* | Xác nhận bằng `nproc` |
| Physical cores | *(điền)* | `lscpu \| grep "Core(s)"` |
| Kiến trúc | x86_64 | |

```bash
lscpu
nproc
```

---

## RAM

| Hạng mục | Thông số | Ghi chú |
|---|---|---|
| Dung lượng | *(vd. ~251 GiB nếu htop hiện 251G)* | Xác nhận `free -h` |
| Swap | *(điền)* | `free -h` |

```bash
free -h
cat /proc/meminfo | head -5
```

---

## Lưu trữ / filesystem

| Hạng mục | Thông số | Ghi chú |
|---|---|---|
| `$HOME` / project | Thường **NFS** (`nfs4`) | Chậm I/O train nếu đọc ảnh trực tiếp |
| Local nhanh | `/tmp` hoặc SSD local | Nên stage dataset vào đây |
| Dung lượng trống | *(điền `df -h`)* | |

```bash
df -hT
df -T $HOME /tmp
```

---

## Mạng (nếu cần)

| Hạng mục | Thông số |
|---|---|
| Hostname | *(vd. SERVER-03 / CPU-FPGA-GPU)* |
| NFS server | *(vd. từ `df -T` — IP mount)* |

```bash
hostname
hostnamectl
```

---

## Bảng tóm tắt dùng cho paper / phụ lục

| Component | Specification |
|---|---|
| GPUs | 2 × NVIDIA GeForce RTX 3090, 24 GB VRAM each |
| CPU | *(fill)* |
| System memory | *(fill)* |
| OS | Ubuntu Linux (x86_64) |
| Storage notes | Project data may reside on NFS; training images staged to local disk when possible |

---

## Thu thập đủ thông số (chạy 1 lần trên server)

```bash
echo "=== HOST ===" && hostname && uname -a
echo "=== CPU ===" && lscpu | egrep "Model name|Socket|Core|Thread|CPU\(s\)|MHz"
echo "=== RAM ===" && free -h
echo "=== GPU ===" && nvidia-smi -L && nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.max_limit --format=csv
echo "=== DISK ===" && df -hT | head -20
```

Copy output vào các ô *(điền)* ở trên khi cần bản final cho paper.
