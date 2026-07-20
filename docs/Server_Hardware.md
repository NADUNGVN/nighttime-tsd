# Server train — thông số phần cứng

| Hạng mục | Giá trị |
|---|---|
| Hostname | CPU-FPGA-GPU / SERVER-03 |
| OS | Ubuntu Linux x86_64 |
| **CPU** | Intel Core **i7-8700K** @ 3.7 GHz |
| Cores / threads | **6 physical / 12 logical** |
| L3 | 12 MiB |
| **RAM** | **~32 GB** (MemTotal ≈ 31 GiB) + Swap 2 GiB |
| **GPU** | **NVIDIA RTX 3090** 24 GB VRAM |
| GPU power | max ~350 W |
| CUDA (driver guide) | hỗ trợ CUDA 12.x |
| **Storage data** | `$HOME` thường mount **NFS** (`nfs4`) — **nút thắt chính** |

## Pipeline train (đúng thực tế)

```
NFS (ảnh JPG)  →  CPU dataloader + augment  →  PCIe  →  3090 VRAM
     ↑ bottleneck #1      ↑ bottleneck #2 nếu workers=0
```

| Thành phần | Vai trò | Giới hạn trên máy này |
|---|---|---|
| NFS | Đọc 16k ảnh CCTSDB | Latency cao, song song 2 job càng chậm |
| CPU 6C | Decode JPEG, augment, collate | `workers=0` → 1 core nuôi GPU → **GPU util ~0%** |
| RAM 32 GB | Cache nhãn + (tuỳ) cache ảnh | `cache=ram` full 16k chỉ an toàn sau khi data **local** |
| 3090 24 GB | Forward/backward | Nano batch 64–128 **dư VRAM** nếu được feed đủ |

Chi tiết tối ưu: [`BOTTLENECK_TRAIN_3090.md`](BOTTLENECK_TRAIN_3090.md).
