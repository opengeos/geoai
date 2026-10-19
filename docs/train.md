# train module

Provides comprehensive model training pipelines for object detection, instance segmentation, and semantic segmentation on geospatial imagery. This module handles dataset preparation, augmentation, training loops, evaluation, and model export, supporting architectures such as Mask R-CNN, Faster R-CNN, and U-Net variants with geospatial-aware data loading.

Pass `k_folds=5` to `train_segmentation_model` for built-in CV with fresh models,
training on K-1 folds and validation on the remaining fold. That held-out fold
also selects the checkpoint and supplies OOF predictions; `val_split` is ignored
in CV mode. Each fold retains the usual history/checkpoint format for
`plot_performance_metrics` and `semantic_segmentation_batch`. See
[K-fold cross-validation](cross_validation.md) for built-in and callback-based usage.
Choose `monitor="val_loss"` or `monitor="val_iou"` for checkpoint selection and
early stopping. Pass the returned `ensemble_model_paths` to batch inference to
average selected fold models' probabilities on new imagery.

!!! example "Related Examples"
    - [Train Object Detection Model](examples/train_object_detection_model.ipynb)
    - [Train Segmentation Model](examples/train_segmentation_model.ipynb)
    - [K-fold Cross-validation](examples/kfold_cross_validation.ipynb)
    - [Train Instance Segmentation Model](examples/train_instance_segmentation_model.ipynb)

::: geoai.train
