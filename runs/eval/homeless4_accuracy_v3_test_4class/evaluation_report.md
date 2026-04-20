# Evaluation Report

- Dataset: `/Users/spalash/Documents/BDA600/StreetBridge/datasets/yolo_homeless_4class`
- Split: `test`
- Images: 47
- Run directory: `/Users/spalash/Documents/BDA600/StreetBridge/runs/detect/homeless4_accuracy_v3`
- Evaluation directory: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_accuracy_v3_test_4class`

## Training Summary

- Best epoch by mAP50-95: 29
- Best precision: 0.4601
- Best recall: 0.5232
- Best mAP50: 0.4924
- Best mAP50-95: 0.2164

## Ground-Truth Support

- homeless_tent: 86 boxes
- homeless_cart: 33 boxes
- homeless_bicycle: 6 boxes
- homeless_person: 38 boxes

## Support Notes

- homeless_bicycle: very low support (6 boxes)

## Key Artifacts

- confusion_matrix: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_accuracy_v3_test_4class/confusion_matrix.png`
- confusion_matrix_normalized: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_accuracy_v3_test_4class/confusion_matrix_normalized.png`
- pr_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_accuracy_v3_test_4class/BoxPR_curve.png`
- precision_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_accuracy_v3_test_4class/BoxP_curve.png`
- recall_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_accuracy_v3_test_4class/BoxR_curve.png`
- f1_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless4_accuracy_v3_test_4class/BoxF1_curve.png`

## Visual Error Analysis

- No paired prediction preview images were found.

## What To Check

- Compare confusion matrices to see which classes are being confused.
- Inspect PR, precision, recall, and F1 curves for threshold behavior.
- Review the prediction preview images for repeated false positives and misses.
- Pay special attention to low-support classes because their metrics will be noisy.
