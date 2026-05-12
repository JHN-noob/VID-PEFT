# VID-PEFT Publication Tables

Tables are aggregate rows built from completed local run artifacts. Table 1 combines seed 0/1/2 as mean +/- std. Tables 2-4 are currently seed0-only ablations and are formatted with n=1.

## Table 1. Main Results Across Seeds

| model | condition | metric | best | final | train_time_hms_mean | trainable_params | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-DETRv2-S | Head-only | AP | 0.3240 +/- 0.0097 (n=3) | 0.3117 +/- 0.0093 (n=3) | 0:19:17 | 51616 (n=3) | 0,1,2 |
| RT-DETRv2-S | Spatial full FT | AP | 0.3585 +/- 0.0170 (n=3) | 0.3250 +/- 0.0022 (n=3) | 0:42:15 | 20133104 (n=3) | 0,1,2 |
| RT-DETRv2-S | Spatial+Temporal PEFT | AP | 0.3385 +/- 0.0075 (n=3) | 0.2693 +/- 0.0098 (n=3) | 1:29:16 | 24632 (n=3) | 0,1,2 |
| YOLOv8m | Head-only | mAP50-95 | 0.3965 +/- 0.0094 (n=3) | 0.3944 +/- 0.0122 (n=3) | 0:08:45 | 3798840 (n=3) | 0,1,2 |
| YOLOv8m | Spatial full FT | mAP50-95 | 0.3923 +/- 0.0077 (n=3) | 0.3923 +/- 0.0077 (n=3) | 0:24:22 | 25879464 (n=3) | 0,1,2 |
| YOLOv8m | Spatial+Temporal PEFT | mAP50-95 | 0.3983 +/- 0.0121 (n=3) | 0.3971 +/- 0.0119 (n=3) | 0:25:46 | 24800 (n=3) | 0,1,2 |

## Table 2. Budget Sweep

| model | budget | metric | best | final | train_time_hms_mean | trainable_params | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-DETRv2-S | small | AP | 0.3295 +/- 0.0000 (n=1) | 0.2394 +/- 0.0000 (n=1) | 1:41:20 | 12316 (n=1) | 0 |
| RT-DETRv2-S | medium | AP | 0.3287 +/- 0.0000 (n=1) | 0.2649 +/- 0.0000 (n=1) | 1:30:31 | 24632 (n=1) | 0 |
| RT-DETRv2-S | large | AP | 0.3379 +/- 0.0000 (n=1) | 0.2620 +/- 0.0000 (n=1) | 1:42:11 | 49264 (n=1) | 0 |
| YOLOv8m | small | mAP50-95 | 0.3974 +/- 0.0000 (n=1) | 0.3951 +/- 0.0000 (n=1) | 0:27:04 | 12400 (n=1) | 0 |
| YOLOv8m | medium | mAP50-95 | 0.3977 +/- 0.0000 (n=1) | 0.3954 +/- 0.0000 (n=1) | 0:26:26 | 24800 (n=1) | 0 |
| YOLOv8m | large | mAP50-95 | 0.3966 +/- 0.0000 (n=1) | 0.3936 +/- 0.0000 (n=1) | 0:26:04 | 49600 (n=1) | 0 |

## Table 3. Clip Sweep

| model | clip | metric | best | final | train_time_hms_mean | trainable_params | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-DETRv2-S | T1_offline | AP | 0.3227 +/- 0.0000 (n=1) | 0.2262 +/- 0.0000 (n=1) | 0:34:23 | 24632 (n=1) | 0 |
| RT-DETRv2-S | T3_offline | AP | 0.3173 +/- 0.0000 (n=1) | 0.2339 +/- 0.0000 (n=1) | 1:00:57 | 24632 (n=1) | 0 |
| RT-DETRv2-S | T5_offline | AP | 0.3287 +/- 0.0000 (n=1) | 0.2649 +/- 0.0000 (n=1) | 1:30:31 | 24632 (n=1) | 0 |
| RT-DETRv2-S | T7_offline | AP | 0.3232 +/- 0.0000 (n=1) | 0.2483 +/- 0.0000 (n=1) | 1:59:26 | 24632 (n=1) | 0 |
| RT-DETRv2-S | T5_causal | AP | 0.3268 +/- 0.0000 (n=1) | 0.2353 +/- 0.0000 (n=1) | 1:29:16 | 24632 (n=1) | 0 |
| YOLOv8m | T1_offline | mAP50-95 | 0.3978 +/- 0.0000 (n=1) | 0.3967 +/- 0.0000 (n=1) | 0:16:35 | 24800 (n=1) | 0 |
| YOLOv8m | T3_offline | mAP50-95 | 0.3979 +/- 0.0000 (n=1) | 0.3961 +/- 0.0000 (n=1) | 0:19:10 | 24800 (n=1) | 0 |
| YOLOv8m | T5_offline | mAP50-95 | 0.3977 +/- 0.0000 (n=1) | 0.3954 +/- 0.0000 (n=1) | 0:26:26 | 24800 (n=1) | 0 |
| YOLOv8m | T7_offline | mAP50-95 | 0.3980 +/- 0.0000 (n=1) | 0.3954 +/- 0.0000 (n=1) | 0:32:21 | 24800 (n=1) | 0 |
| YOLOv8m | T5_causal | mAP50-95 | 0.3974 +/- 0.0000 (n=1) | 0.3948 +/- 0.0000 (n=1) | 0:25:24 | 24800 (n=1) | 0 |

## Table 4. Frame Stability

| model | clip | matched_iou_mean | unmatched_rate_mean | center_shift_norm_mean | train_time_hms_mean | trainable_params | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-DETRv2-S | T1_offline | 0.7524 | 0.3736 | 0.01736 | 0:34:23 | 24632 (n=1) | 0 |
| RT-DETRv2-S | T3_offline | 0.7390 | 0.3938 | 0.01650 | 1:00:57 | 24632 (n=1) | 0 |
| RT-DETRv2-S | T5_offline | 0.7757 | 0.3341 | 0.01595 | 1:30:31 | 24632 (n=1) | 0 |
| RT-DETRv2-S | T7_offline | 0.7837 | 0.3262 | 0.01602 | 1:59:26 | 24632 (n=1) | 0 |
| RT-DETRv2-S | T5_causal | 0.7665 | 0.3316 | 0.01602 | 1:29:16 | 24632 (n=1) | 0 |
| YOLOv8m | T1_offline | 0.7549 | 0.2834 | 0.01284 | 0:16:35 | 24800 (n=1) | 0 |
| YOLOv8m | T3_offline | 0.7538 | 0.2869 | 0.01293 | 0:19:10 | 24800 (n=1) | 0 |
| YOLOv8m | T5_offline | 0.7538 | 0.2890 | 0.01292 | 0:26:26 | 24800 (n=1) | 0 |
| YOLOv8m | T7_offline | 0.7538 | 0.2879 | 0.01292 | 0:32:21 | 24800 (n=1) | 0 |
| YOLOv8m | T5_causal | 0.7534 | 0.2877 | 0.01294 | 0:25:24 | 24800 (n=1) | 0 |
