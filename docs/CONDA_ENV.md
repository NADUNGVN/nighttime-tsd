# Conda env riêng — không ảnh hưởng AI khác

## Tên env

| Env | Mục đích |
|---|---|
| **`nighttime-tsd`** | Chỉ project night TSD / YOLO train |
| `base` / `ai` / … | **Không đụng** — script không `conda install` vào base |

## Trên server Linux (3090)

```bash
cd ~/nighttime-tsd   # hoặc path clone

# Nếu chưa có conda (1 lần, user-local):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash
# mở shell mới, rồi:

bash scripts/setup_conda_env.sh --with-torch-cu124 --install-htop
conda activate nighttime-tsd
```

### Kiểm tra env hiện có (trước khi tạo)

```bash
conda env list
# Chỉ thêm nighttime-tsd; không xóa/sửa env khác
```

### Công cụ monitor

| Tool | Cài | Ghi chú |
|---|---|---|
| **nvitop** | trong conda env (`nvitop`) | GPU util, VRAM, process |
| **htop** | system (`sudo apt install htop`) | CPU/RAM — không nằm trong conda |

```bash
conda activate nighttime-tsd
nvitop
htop
```

### Xóa env này (không ảnh hưởng env khác)

```bash
conda deactivate
conda env remove -n nighttime-tsd
```

## Máy Windows (laptop) hiện tại

Conda **chưa cài**. Có thể:

1. Train trên **server 3090** (khuyến nghị) → dùng script trên, hoặc  
2. Cài Miniconda Windows: https://docs.conda.io/en/latest/miniconda.html  
   rồi: `conda env create -f environment.yml`

## Load model trong env

```bash
conda activate nighttime-tsd
python scripts/load_model.py --model yolo11n.pt --info
python scripts/train_baseline.py --model yolo11n.pt --batch 64 --epochs 100 --name yolo11n_cntsss
```
