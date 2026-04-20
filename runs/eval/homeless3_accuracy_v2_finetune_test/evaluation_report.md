# Evaluation Report

- Dataset: `/Users/spalash/Documents/BDA600/StreetBridge/datasets/yolo_homeless_4class_clean_balanced`
- Split: `test`
- Images: 94
- Run directory: `/Users/spalash/Documents/BDA600/StreetBridge/runs/detect/homeless3_accuracy_v2_finetune`
- Evaluation directory: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless3_accuracy_v2_finetune_test`

## Training Summary

- Best epoch by mAP50-95: 22
- Best precision: 0.4843
- Best recall: 0.4553
- Best mAP50: 0.4376
- Best mAP50-95: 0.1890

## Ground-Truth Support

- homeless_tent: 229 boxes
- homeless_cart: 43 boxes
- homeless_person: 50 boxes

## Support Notes

- All classes have at least 25 ground-truth boxes in this split.

## Key Artifacts

- confusion_matrix: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless3_accuracy_v2_finetune_test/confusion_matrix.png`
- confusion_matrix_normalized: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless3_accuracy_v2_finetune_test/confusion_matrix_normalized.png`
- pr_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless3_accuracy_v2_finetune_test/BoxPR_curve.png`
- precision_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless3_accuracy_v2_finetune_test/BoxP_curve.png`
- recall_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless3_accuracy_v2_finetune_test/BoxR_curve.png`
- f1_curve: `/Users/spalash/Documents/BDA600/StreetBridge/runs/eval/homeless3_accuracy_v2_finetune_test/BoxF1_curve.png`

## Visual Error Analysis

- No paired prediction preview images were found.

## What To Check

- Compare confusion matrices to see which classes are being confused.
- Inspect PR, precision, recall, and F1 curves for threshold behavior.
- Review the prediction preview images for repeated false positives and misses.
- Pay special attention to low-support classes because their metrics will be noisy.
