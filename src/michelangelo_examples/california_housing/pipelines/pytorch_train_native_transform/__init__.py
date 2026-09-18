"""California Housing example: native PyTorch transforms via tabular_native_transform.

Extends the ``pytorch_train`` pipeline with a ``tabular_native_transform``
stage that applies PyTorch-layer feature transforms (scale, log, clip) on
Ray, demonstrating the same transform DAG design used at both training and
serving time in full Michelangelo pipelines.
"""
