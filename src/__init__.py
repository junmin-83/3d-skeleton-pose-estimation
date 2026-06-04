"""3D skeleton pose estimation from 2 RGB + 1 RGB-D camera.

Hybrid pipeline: multi-view confidence-weighted triangulation resolves the
monocular depth ambiguity; the depth sensor corrects scale and fills occluded
joints. See docs/SPEC.md for the full specification.
"""
