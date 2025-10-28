"""Benchmark pipeline shared utilities for multi-model 3D human pose evaluation.

This package provides:
- metrics: MPJPE and PA-MPJPE consistent across models
- pipeline: contract and runner to orchestrate data -> weights -> inference -> metrics -> report
- models: thin wrappers for external models (RootNet, MobileHumanPose, Integral Human Pose)
- report: JSON/Markdown/plots consolidation

All functions are CPU/GPU agnostic and accept a `device` parameter when relevant.
"""
