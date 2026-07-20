# Gỡ nghẽn cổ chai train YOLO trên 3090 (i7-8700K + 32GB + NFS)

## Chẩn đoán (từ nvitop / log thực tế)

| Triệu chứng | Nguyên nhân |
|---|---|
| GPU-Util **0%**, process train vẫn sống, CPU process **~90%+** | Dataloader **`workers=0`**: 1 thread đọc NFS + decode → GPU đói data |
| Power ~150–180W, MEM GPU ~9 GB nhưng util thấp | GPU đang **chờ batch**, không phải thiếu VRAM |
| Scan 16k lâu / treo `labels.cache` | NFS + ThreadPool cache nhãn |
| Song song 2 model | Tranh NFS + CPU 6 nhân → **tổng thời gian thường tệ hơn** chạy nối tiếp |

**Kết luận:** nút thắt **không phải 3090**, mà **I/O NFS + CPU feed**.

---

## Chiến lược (theo hiệu quả)

### 1) Bắt buộc — đưa dataset ra **local disk** (tác động lớn nhất)

```bash
# Ưu tiên ổ local (không phải NFS). Thử lần lượt:
df -h /tmp /var/tmp $HOME
df -T /tmp   # cần không phải nfs

# Stage (script repo)
python scripts/stage_data_local.py \
  --src data/processed/cctsdb2021_full \
  --dst /tmp/cctsdb2021_full

# Train trỏ yaml local
python scripts/train_3090.py --model yolo11n.pt --local-data /tmp/cctsdb2021_full
```

`/tmp` trên nhiều máy là local SSD/HDD — **nhanh hơn NFS rất nhiều**.  
Nếu `/tmp` cũng NFS, hỏi admin đường local (vd. `/data/local/$USER`).

### 2) Sau khi data local — tăng workers + (tuỳ) cache

| Setting | NFS (hiện tại) | Local disk |
|---|---|---|
| `workers` | 0–2 | **4–6** (máy 6C/12T; để 2 thread OS) |
| `cache` | False | **`ram`** nếu RAM đủ (~16k JPG thường 3–8 GB) hoặc `disk` |
| `batch` nano | 64 | **96–128** (3090 24GB còn dư) |
| 2 model song song | Không | Vẫn **không** khuyến nghị (1 GPU) |

### 3) Không làm

- 2 train YOLO cùng lúc trên 1 GPU + NFS  
- `workers=16` trên CPU 6 nhân  
- `cache=ram` khi data vẫn trên NFS (lâu + dễ đầy RAM 32GB)

---

## Preset khuyến nghị máy này

**A. Data còn trên NFS (tạm):**
```bash
python scripts/train_3090.py --model yolo11n.pt --preset nfs
# → workers=2, cache=False, batch=64, NUM_THREADS cẩn thận
```

**B. Data đã stage local (tối ưu):**
```bash
python scripts/train_3090.py --model yolo11n.pt --local-data /tmp/cctsdb2021_full --preset local
# → workers=6, cache=ram, batch=96
```

**C. Resume job đang chạy:** không kill giữa epoch trừ khi cần; job hiện tại để xong, job sau dùng preset local.

---

## Kiểm tra hết nghẽn chưa

```bash
nvitop
# Kỳ vọng khi train ổn:
#   GPU-Util 70–99% phần lớn thời gian
#   không còn 0% kéo dài hàng chục giây giữa các batch
```

```bash
# So tốc độ đọc
python scripts/preflight_cctsdb.py --data-root /tmp/cctsdb2021_full --read-n 100
# img/s local phải cao hơn rõ so với path NFS
```
