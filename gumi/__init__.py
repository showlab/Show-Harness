"""GUMI: the GUI-based Manipulation Interface (paper section 4).

Maps each semantic action unit to a key/button so humans and agents demonstrate
tasks by directly playing the robot -- no teleoperation hardware. Every step is
recorded as a ready (observation, action) training pair in the same layout the
fine-tuned policy consumes.

Collection modes:
  * Human, browser:   collect_rollouts_web.py  (single arm, :8600)
                      collect_rollouts_web_dual.py (dual-arm compose-and-commit, :8620)
  * Agent, HTTP:      gpt_web_operator.py (:8630) -- an operator model plays either
                      through /api/step and a polling bridge.
  * Agent, computer-use: hand prompts/web_operator*.txt to a GUI-driving agent
                      pointed at :8600 or :8620.
  * Human, local:     core/teleop*.py pygame collectors (scripts/trajectory/).
  * Intervention:     plugins/dagger -- the same keys override a live rollout.
"""
