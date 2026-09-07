"""RoboLab (Isaac Lab) side of the atomic-action data generation.

Sibling of ``real2sim/maniskill/``. The simulator-agnostic core lives in
``real2sim/atomic_tokenizer.py``; the only RoboLab-aware layers are
``real2sim/backends/robolab.py`` (the AtomicSimEnv implementation) and this package's
``tasks.py`` (plan + geometry) / ``oracle.py`` (Scheme A generator).
"""
