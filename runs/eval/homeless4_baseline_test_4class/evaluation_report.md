# Evaluation Report

- Dataset: `/Users/spalash/Documents/BDA600/StreetBridge/datasets/yolo_homeless_4class`
- Split: `test`
- Images: 47
- Run directory: `/Users/spalash/Documents/BDA600/StreetBridge/runs/detect/runs/detect/homeless4_baseline`
- Evaluation directory: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_baseline_test_4class`

## Training Summary

- Best epoch by mAP50-95: 15
- Best precision: 0.5323
- Best recall: 0.5212
- Best mAP50: 0.5065
- Best mAP50-95: 0.2258

## Ground-Truth Support

- homeless_tent: 86 boxes
- homeless_cart: 33 boxes
- homeless_bicycle: 6 boxes
- homeless_person: 38 boxes

## Support Notes

- homeless_bicycle: very low support (6 boxes)

## Key Artifacts

- confusion_matrix: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_baseline_test_4class/confusion_matrix.png`
- confusion_matrix_normalized: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_baseline_test_4class/confusion_matrix_normalized.png`
- pr_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_baseline_test_4class/BoxPR_curve.png`
- precision_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_baseline_test_4class/BoxP_curve.png`
- recall_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_baseline_test_4class/BoxR_curve.png`
- f1_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_baseline_test_4class/BoxF1_curve.png`

## Visual Error Analysis

- No paired prediction preview images were found.

## What To Check

- Compare confusion matrices to see which classes are being confused.
- Inspect PR, precision, recall, and F1 curves for threshold behavior.
- Review the prediction preview images for repeated false positives and misses.
- Pay special attention to low-support classes because their metrics will be noisy.
