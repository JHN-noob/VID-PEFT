# Outputs Index

`outputs/`는 원본 YouTube-VIS 데이터를 수정하지 않고 생성한 파생 데이터, config snapshot, 학습 결과, publication 산출물을 보관합니다.

## 보존 기준

현재 publication 기준 active 산출물을 보존합니다.

- active run 30개
- active config snapshot 30개
- RT-DETR generated YAML 15개
- seed repeat summary
- publication table/figure
- frame-stability prediction/report
- YouTube-VIS detection 파생 manifest/split/export

## 디렉터리 구조

```text
outputs/
  manifests/
  splits/
  yolo_datasets/
  rtdetr_datasets/
  rtdetr_configs/
  main_configs/
  runs/
  frame_stability/
  summaries/
```

## 파생 데이터

원본 데이터는 `data/VIS2021`에 유지하고, 아래 파생 산출물만 `outputs/`에 둡니다.

- `manifests/youtube_vis.jsonl`
- `splits/youtube_vis_pilot_train.jsonl`
- `splits/youtube_vis_pilot_dev.jsonl`
- `splits/youtube_vis_train.jsonl`
- `splits/youtube_vis_val.jsonl`
- `yolo_datasets/youtube_vis_pilot/`
- `rtdetr_datasets/youtube_vis_pilot/`

`yolo_datasets/`와 `rtdetr_datasets/`는 학습 재현용 파생 export입니다.

## Config 산출물

- `main_configs/`: active experiment config snapshot 30개
- `rtdetr_configs/`: official RT-DETR runner가 읽는 generated YAML 15개

`configs/`의 root-level JSON은 seed0 기준 template이고, `outputs/main_configs/`는 실제 active run별 expanded snapshot입니다.

## Active Runs

`runs/`에는 active matrix에 대응하는 30개 run을 남깁니다.

| 그룹 | 수량 | 설명 |
|---|---:|---|
| Head warmup | 6 | YOLO/RT-DETR x seed 0/1/2 |
| Main comparison | 12 | YOLO/RT-DETR x spatial full FT/spatial+temporal PEFT x seed 0/1/2 |
| Budget sweep | 4 | small/large 추가 run, medium은 main T5 anchor 재사용 |
| Clip sweep | 8 | T1/T3/T7/causal T5 추가 run, offline T5는 main anchor 재사용 |

각 run은 아래 파일을 포함합니다.

- `config_snapshot.json`
- `metrics.json`
- `policy_report.json`
- model checkpoint 또는 official trainer log

## Frame Stability

Frame stability는 best checkpoint에서 per-frame prediction JSONL을 생성한 뒤 계산합니다.

- `frame_stability/frame_stability_specs.json`
- `frame_stability/predictions/{experiment_id}.jsonl`
- `frame_stability/reports/{experiment_id}_stability.json`
- `frame_stability/reports/{experiment_id}_stability.csv`
- `frame_stability/reports/frame_stability_summary.json`
- `frame_stability/reports/frame_stability_summary.csv`

Table 4/Figure 4는 이 디렉터리의 report를 사용해 재생성합니다.

## Summaries

최종 보고용 산출물은 `summaries/publication/`입니다.

```text
summaries/
  seed_repeats/
  publication/
    tables/
    figures/
```

Seed repeat summary:

- `summaries/seed_repeats/seed_repeat_runs.csv`
- `summaries/seed_repeats/seed_repeat_summary.csv`
- `summaries/seed_repeats/seed_repeat_summary.json`

Publication tables:

- `summaries/publication/tables/table1_main_seed_combined.csv`
- `summaries/publication/tables/table2_budget_seed_combined.csv`
- `summaries/publication/tables/table3_clip_seed_combined.csv`
- `summaries/publication/tables/table4_stability_seed_combined.csv`
- `summaries/publication/tables/publication_tables.md`
- `summaries/publication/tables/publication_tables.json`

Publication figures:

- `summaries/publication/figures/figure1_main_best_final.png`
- `summaries/publication/figures/figure2_budget_sweep.png`
- `summaries/publication/figures/figure3_clip_sweep.png`
- `summaries/publication/figures/figure4_frame_stability.png`

## 해석 주의

- Table 1은 seed 0/1/2 평균이므로 main conclusion에 사용합니다.
- Table 2-4는 seed0 ablation입니다. 경향 분석으로 사용하고 통계적 일반화는 제한합니다.
- YOLO와 RT-DETR의 primary metric은 서로 다릅니다. YOLO는 mAP50-95, RT-DETR은 COCO AP입니다.
- 모델 간 직접 수치 비교보다 조건 변화에 대한 모델 내부 반응과 best-final gap을 우선 해석합니다.
