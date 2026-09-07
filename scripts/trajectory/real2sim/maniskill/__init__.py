"""ManiSkill data generation: task construction (``tasks.py``) + the two closed-loop
generators (``oracle.py`` = Scheme A, ``record_demos.py`` + ``follow_tokenize.py`` = Scheme D)
+ dataset tooling (``make_dataset.py``, ``merge_shards.py``; previews via ``../preview.py``).

The scenes themselves live in ``core/`` (``maniskill_task`` + the ``maniskill_scenes``
table) because the deployment runner builds its env from exactly those modules --
generating data from a private copy would let train and deploy drift apart. This package
wraps them, it does not redefine them: ``tasks.py`` describes only LAYOUT SAMPLING, and
takes each task's instruction and robot uid straight from the deployment table.
"""
