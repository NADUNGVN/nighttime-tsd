# External datasets: TT100K, MTSD, and CURE-TSD

This project may add three large external sources, but none replaces the
CCTSDB2021 official test.  They are raw-data inputs outside Git; only source
URLs, access records, scripts, compact inventories, and approved results may
be committed.

| Source | Acquisition | Planned use | Mandatory guardrail |
| --- | --- | --- | --- |
| TT100K | Direct download from the official Tsinghua server under CC BY-NC. | External pretraining or pretraining then CCTSDB fine-tuning. | Preserve the original archive/annotations; audit the 221-category taxonomy and grouping before mapping classes. |
| MTSD | Download from the official Mapillary dataset page only after accepting its research-use license. | Fully annotated subset for external pretraining or pretraining then CCTSDB fine-tuning. | Exclude partially annotated images; do not publish images or derived labels. |
| CURE-TSD | Request/download via the official OLIVES/GaTech release and accept its terms. | Controlled robustness/adaptation data. | Use detection frames plus bounding boxes, retain sequence and challenge metadata, and split by sequence before selecting frames. |

The official sources and access terms are frozen in
`configs/external_datasets_v1.json`. TT100K is CC BY-NC according to its
official page. The MTSD research-use license does not
permit redistribution of the dataset or derivative labels/images.  For
CURE-TSD, retain the terms received with the release and do not assume that
the code repository itself grants rights to the data.

## Server layout

Keep raw releases in the existing data checkout, not in
`nighttime-tsd-new`.  The following command creates only empty directories:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && mkdir -p ../nighttime-tsd/data/raw/TT100K/incoming ../nighttime-tsd/data/raw/TT100K/release ../nighttime-tsd/data/raw/TT100K/checksums ../nighttime-tsd/data/raw/MTSD/incoming ../nighttime-tsd/data/raw/MTSD/release ../nighttime-tsd/data/raw/MTSD/checksums ../nighttime-tsd/data/raw/CURE-TSD/incoming ../nighttime-tsd/data/raw/CURE-TSD/release ../nighttime-tsd/data/raw/CURE-TSD/checksums
```

TT100K can be downloaded directly from the official Tsinghua server. The
following one-line command is resumable, records a checksum, and extracts to
the separate release directory. Check that the server has at least 100 GB free
before running it, as the official tutorial recommends.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && df -h ../nighttime-tsd/data/raw && wget -c -O ../nighttime-tsd/data/raw/TT100K/incoming/data.zip https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip && sha256sum ../nighttime-tsd/data/raw/TT100K/incoming/data.zip | tee ../nighttime-tsd/data/raw/TT100K/checksums/data.zip.sha256 && unzip -q ../nighttime-tsd/data/raw/TT100K/incoming/data.zip -d ../nighttime-tsd/data/raw/TT100K/release
```

MTSD and CURE-TSD require their corresponding official terms to be accepted;
place their official archive(s) in the relevant `incoming/` directory only
after that step. Do not use an unofficial mirror, Kaggle repackaging, or a
scraped image set for this paper.

After receiving an archive, replace `/path/to/official-file.zip` below with
the exact local file, record its SHA-256, and extract it to the separate
`release/` directory.  Do not overwrite an existing release.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && sha256sum /path/to/official-file.zip | tee ../nighttime-tsd/data/raw/MTSD/checksums/official-file.zip.sha256
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && unzip -q /path/to/official-file.zip -d ../nighttime-tsd/data/raw/MTSD/release
```

Use the same commands with `CURE-TSD` in place of `MTSD` for that release.
If an official release is split across several archives, checksum every
archive before extraction and retain their original filenames.

## Required audit before conversion

Run the repository's format-agnostic inventory on the extracted release.  It
does not alter raw files and is intentionally not a data converter.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/inspect_external_dataset.py --dataset tt100k --raw ../nighttime-tsd/data/raw/TT100K/release --out data/external/tt100k_release_inventory.json
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/inspect_external_dataset.py --dataset mtsd --raw ../nighttime-tsd/data/raw/MTSD/release --out data/external/mtsd_release_inventory.json
```

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && python scripts/inspect_external_dataset.py --dataset cure_tsd --raw ../nighttime-tsd/data/raw/CURE-TSD/release --out data/external/cure_tsd_release_inventory.json
```

Push only the two inventory JSON files if their contents do not contain
restricted filenames or private metadata; otherwise share their console count
summary and selected structure with the authors.  We will write a
dataset-specific converter only after verifying the actual release layout,
class taxonomy, label semantics, and sequence metadata.

## Experimental boundary

The first external-data comparison is selected on CCTSDB development only:
`CCTSDB-only`, `MTSD -> CCTSDB`, `CURE-TSD -> CCTSDB`, and optionally
`MTSD + CURE-TSD -> CCTSDB`.  The CCTSDB official test remains untouched
until this regimen is frozen.  External source images must never be used for
TensorRT calibration in the CCTSDB quantization experiment unless a later,
separately pre-registered experiment explicitly studies that question.
