import os

# Must happen before tensorflow (and therefore magi) is imported anywhere in
# the test session - matches the convention in tools/*.py and every notebook.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
