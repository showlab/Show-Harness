"""Sim-agnostic atomic-action discretiser -- the reusable core of real2sim data generation.

This is the ONE file that owns "what an atomic token means": the vocabulary, the closed-loop
execution that pins one ``MV_*`` token to exactly ``step_m`` metres of end-effector travel,
the planners that turn a continuous target/path into single-axis token runs, and the writer
that stores the result in the real-robot teleop layout. **Nothing in here knows about any
particular simulator.** Swapping ManiSkill for Isaac / a replayed real-robot log
means implementing :class:`AtomicSimEnv` for that backend (see ``backends/``) -- the
discretisation logic below is unchanged.

Why the token has to be executed rather than merely labelled
------------------------------------------------------------
The policy is trained on (frame, token) pairs and deployed by executing one token per
decision, so training frames must sit on the SAME discrete lattice the deployment loop
walks. Labelling a continuous demo offline breaks that: a frame-to-frame move on a smooth
path is diagonal, while its label claims a single axis (measured 65-72% off-axis on this
project's demos -- unusable). Every generator here therefore *executes* the token it is
about to record, closed-loop, and stores the frame observed BEFORE the token runs.

Contract (matches ``configs/primitives_franka.yaml``, the real-Franka convention verified in
deployment): ``MV_FWD``=+X, ``MV_BACK``=-X, ``MV_LEFT``=-Y, ``MV_RIGHT``=+Y, ``MV_UP``=+Z,
``MV_DOWN``=-Z, in world axes (~= robot base axes at identity base pose). One token == one
``step_m`` (2 cm) displacement; ``gripper_closed`` in a record is the state BEFORE the token.
"""
from __future__ import annotations

import abc
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------ vocabulary
MOVE_DIRS: dict[str, np.ndarray] = {
    "MV_FWD": np.array([1.0, 0.0, 0.0]),
    "MV_BACK": np.array([-1.0, 0.0, 0.0]),
    "MV_LEFT": np.array([0.0, -1.0, 0.0]),
    "MV_RIGHT": np.array([0.0, 1.0, 0.0]),
    "MV_UP": np.array([0.0, 0.0, 1.0]),
    "MV_DOWN": np.array([0.0, 0.0, -1.0]),
}
GRASP = "GRASP"
RELEASE = "RELEASE"

OPPOSITE: dict[str, str] = {
    "MV_FWD": "MV_BACK", "MV_BACK": "MV_FWD",
    "MV_LEFT": "MV_RIGHT", "MV_RIGHT": "MV_LEFT",
    "MV_UP": "MV_DOWN", "MV_DOWN": "MV_UP",
}

# Axis index/sign -> token, for planners: token_for_axis(1, -1) == "MV_LEFT".
_AXIS_TOKENS = {
    (0, 1): "MV_FWD",
    (0, -1): "MV_BACK",
    (1, -1): "MV_LEFT",
    (1, 1): "MV_RIGHT",
    (2, 1): "MV_UP",
    (2, -1): "MV_DOWN",
}
# token -> (axis index, sign); the inverse map, handy for stats/validation.
TOKEN_AXIS = {tok: (ax, sg) for (ax, sg), tok in _AXIS_TOKENS.items()}


def token_for_axis(axis: int, sign: float) -> str:
    return _AXIS_TOKENS[(int(axis), 1 if sign >= 0 else -1)]


def is_move(token: str) -> bool:
    return token in MOVE_DIRS


def token_kind(token: str) -> str:
    """"move" | "grasp" | "release" -- the ``kind`` field of a rollout record."""
    if token in MOVE_DIRS:
        return "move"
    if token == GRASP:
        return "grasp"
    if token == RELEASE:
        return "release"
    raise ValueError(f"unknown token {token!r}")


def opposite(a: str, b: str) -> bool:
    return OPPOSITE.get(a) == b


def to_np(t: Any) -> np.ndarray:
    """Torch tensor / array-like -> numpy (sims hand back batched torch tensors)."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# -------------------------------------------------------------- backend contract
class AtomicSimEnv(abc.ABC):
    """Everything the discretiser needs from a simulator -- implement this per sim.

    The backend owns the sim: how an EE displacement command is applied, how the cameras
    are read and transformed into the two deployment views, and how success is judged.
    It does NOT decide token size, planning, or file layout -- those live in this module
    and stay identical across sims.

    ``delta_bound_m`` is the controller's action scale: the metres commanded at action 1.0
    (0.1 m for ManiSkill's ``pd_ee_delta_pos``). :class:`AtomicExec` never exceeds its own
    ``max_cmd_m`` per control step, so this only sets the normalisation.
    """

    delta_bound_m: float = 0.1

    # -- state readback ------------------------------------------------------
    @abc.abstractmethod
    def tcp_pos(self) -> np.ndarray:
        """End-effector position, shape (3,), world/robot-base frame."""

    @abc.abstractmethod
    def tcp_pose7(self) -> list[float]:
        """[x, y, z, qw, qx, qy, qz] -- the real recorder's ``ee_pose`` field."""

    @abc.abstractmethod
    def gripper_width(self) -> float:
        """Finger opening in metres (used to detect an empty grasp / a dropped object)."""

    # -- actuation -----------------------------------------------------------
    @abc.abstractmethod
    def apply_delta(self, delta_m: np.ndarray, grip_cmd: float,
                    max_cmd_m: float) -> None:
        """ONE control step: move the EE by (a capped) ``delta_m``, holding ``grip_cmd``.

        ``grip_cmd`` is +1 open / -1 close, and must persist for the whole step.
        """

    # -- observation / evaluation -------------------------------------------
    @abc.abstractmethod
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """(agentview, wrist) uint8 HWC, with the DEPLOYMENT transforms already applied.

        Training images must be byte-identical to what the deployment runner sends, so
        camera choice, rotation and flips belong here, not in the generators.
        """

    @abc.abstractmethod
    def success(self) -> bool:
        """The task's own success predicate, evaluated on the CURRENT state."""

    # -- optional ------------------------------------------------------------
    def frozen(self) -> bool:
        """Has the sim stopped responding to actions for this episode?

        RoboLab freezes an env the moment it terminates: ``step()`` zeroes the actions and
        the scene holds its final state. Every token emitted after that records a frame
        that is identical to the last one while claiming a 2 cm move -- mislabelled
        training data that no aggregate statistic reveals, because the tokens, the
        episode length and the success flag all look normal. Measured on a 10-episode
        RoboLab batch: 80 of 812 samples (10.2%) were post-termination MV_UP tokens with
        EXACTLY 0.00 mm of travel.

        Default False for sims that keep stepping after success.
        """
        return False

    def reset(self, seed: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def prepared_pair(backend: Any, agentview: np.ndarray,
                  wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply THE camera transform contract to both views, from the backend's own fields.

    One call site for both views and every simulator. The two used to be handled
    differently -- the agentview was letterboxed (by the writer, of all places) while the
    wrist was passed through -- on the reasoning that "wrist frames are already square".
    They are, in the SIM. Every real-robot MVTOKEN set stores a 640x480 wrist letterboxed
    into 256x256, i.e. 256x192 of picture between 32 black rows, and both simulators were
    storing a wrist that filled the whole frame instead: the same scene 1.33x larger than
    anything the policy was trained on, on the view it steers by. Nothing detects that --
    the images are valid, the resolutions match, the statistics are unchanged.

    So the letterbox belongs here, next to the crop and the flip, applied to both views
    from the same four fields. ``prepare_view`` skips any stage whose value is falsy, so a
    camera that genuinely needs nothing still passes through untouched.
    """
    from core.record.images import prepare_view  # noqa: PLC0415 -- keeps PIL off the import path

    return (
        prepare_view(
            agentview,
            rotation_degrees=backend.agentview_rotation_degrees,
            flip=backend.agentview_flip,
            crop_aspect=getattr(backend, "agentview_crop_aspect", None),
            square_size=getattr(backend, "agentview_square_size", None),
        ),
        prepare_view(
            wrist,
            rotation_degrees=backend.wrist_rotation_degrees,
            flip=backend.wrist_flip,
            crop_aspect=getattr(backend, "wrist_crop_aspect", None),
            square_size=getattr(backend, "wrist_square_size", None),
        ),
    )


# ------------------------------------------------------------------- execution
class AtomicExec:
    """Execute atomic tokens as accurate ~``step_m`` displacements, closed-loop.

    ``move`` commands the REMAINING error each control step (capped at ``max_cmd_m``)
    until within ``tol_m`` of the target or the step budget runs out. This pins the
    physical meaning of one token to ``step_m`` metres regardless of PD lag, which is the
    whole point of the dataset.

    Gripper tokens first bring the arm to REST (:meth:`quiesce`), then hold the command
    for a fixed number of steps and report the settled width. Acting on the gripper while
    the hand is still travelling is never right: a release hands the object the leftover
    velocity, and a grasp drags the object before the fingers close. Neither shows up as a
    bad token -- the sequence, the step sizes and the gripper width all look correct, and
    only the object's final pose gives it away -- so it is enforced here for every
    simulator and every generator rather than patched per call site.
    """

    def __init__(
        self,
        backend: AtomicSimEnv,
        step_m: float = 0.02,
        tol_m: float = 0.001,
        max_cmd_m: float = 0.02,
        max_ctrl_steps: int = 16,
        gripper_steps: int = 10,
        quiesce_tol_m: float = 0.0003,
        quiesce_max_steps: int = 40,
    ) -> None:
        self.backend = backend
        self.step_m = float(step_m)
        self.tol_m = float(tol_m)
        self.max_cmd_m = float(max_cmd_m)
        self.max_ctrl_steps = int(max_ctrl_steps)
        self.gripper_steps = int(gripper_steps)
        # "Stopped" for :meth:`quiesce`: per-control-step TCP displacement. A token in
        # flight covers ~step_m over max_ctrl_steps, i.e. well over a millimetre per step,
        # so 0.3 mm is comfortably below moving and above numerical noise.
        self.quiesce_tol_m = float(quiesce_tol_m)
        self.quiesce_max_steps = int(quiesce_max_steps)
        self.grip_cmd = 1.0  # +1 open / -1 close, persists across moves

    @property
    def gripper_closed(self) -> bool:
        return self.grip_cmd < 0

    def _step(self, delta_m: np.ndarray) -> None:
        self.backend.apply_delta(np.asarray(delta_m, dtype=np.float64),
                                 self.grip_cmd, self.max_cmd_m)

    def move(self, token: str) -> float:
        """One MV_* token == one ``step_m`` displacement. Returns achieved metres."""
        direction = MOVE_DIRS[token]
        start = self.backend.tcp_pos()
        target = start + direction * self.step_m
        for _ in range(self.max_ctrl_steps):
            err = target - self.backend.tcp_pos()
            if np.linalg.norm(err) < self.tol_m:
                break
            self._step(err)
        return float(np.linalg.norm(self.backend.tcp_pos() - start))

    def move_to(self, target: np.ndarray, tol_m: float = 0.002,
                budget: int = 40) -> None:
        """Servo (not tokenised) -- only for pre-episode positioning, never recorded."""
        for _ in range(budget):
            err = np.asarray(target, dtype=np.float64) - self.backend.tcp_pos()
            if np.linalg.norm(err) < tol_m:
                break
            self._step(err)

    def quiesce(self, tol_m: Optional[float] = None,
                max_steps: Optional[int] = None) -> int:
        """Hold until the TCP stops moving. Returns the control steps used.

        A gripper token must never act while the arm is still travelling, which is what
        :meth:`grasp` and :meth:`release` use this for. Holding is deliberately NOT a
        token: it records no frame and adds no label, so the dataset is unchanged apart
        from the physics being correct.

        Measured rather than a fixed count on purpose -- the settling time is a property
        of the simulator's controller, and the two differ by an order of magnitude
        (ManiSkill's ``pd_ee_delta_pos`` converges in a couple of steps; RoboLab's
        relative IK lags for a dozen). A fixed number would be wrong for one of them.
        """
        tol = self.quiesce_tol_m if tol_m is None else float(tol_m)
        budget = self.quiesce_max_steps if max_steps is None else int(max_steps)
        prev = self.backend.tcp_pos()
        for i in range(budget):
            self._step(np.zeros(3))
            cur = self.backend.tcp_pos()
            if float(np.linalg.norm(cur - prev)) < tol:
                return i + 1
            prev = cur
        return budget

    def grasp(self) -> float:
        # Stop first: closing on an object while the hand is still translating drags or
        # flicks it before the fingers meet, so the grasp lands off-centre or misses.
        self.quiesce()
        self.grip_cmd = -1.0
        for _ in range(self.gripper_steps):
            self._step(np.zeros(3))
        return self.backend.gripper_width()

    def release(self) -> float:
        # Stop first, for a sharper reason: whatever velocity the hand still has is handed
        # to the object. Measured on RoboLab's RubiksCubeTask -- the token before RELEASE
        # was a lateral MV_FWD (the XY drift that builds up during the descent, corrected
        # as one full 2 cm token), so the cube was let go moving forwards; it hit the near
        # rim, knocked the bowl 3.7 cm and bounced 13.7 cm clear of it. Nothing in the
        # episode looked wrong -- the tokens were clean, the displacement statistics were
        # in range, the gripper width was right -- only the object's final position showed
        # it, which is why this is enforced here rather than left to each generator.
        self.quiesce()
        self.grip_cmd = 1.0
        for _ in range(self.gripper_steps):
            self._step(np.zeros(3))
        return self.backend.gripper_width()

    def hold(self, n: int = 1) -> None:
        for _ in range(n):
            self._step(np.zeros(3))


# ------------------------------------------------------------------ recording
# Servo step cap for continuous demos: smaller than the atomic executor's, so the recorded
# path comes out smooth rather than stepped.
SERVO_MAX_CMD_M = 0.015


class DemoRecorder:
    """Drive the sim with CONTINUOUS servo motion, capturing every control step's TCP.

    Input stage of Scheme D: this records a smooth, multi-axis path -- the same character a
    motion planner or a human demo produces -- and :func:`follow_track` later RE-EXECUTES
    that path as single-axis ``step_m`` tokens. Nothing recorded here is ever used as a
    training frame; only the path SHAPE and the gripper event positions survive, which is
    precisely why the follower's frames and labels cannot disagree.

    Sim-agnostic: it only ever calls ``tcp_pos`` / ``apply_delta`` / ``success`` on the
    backend, so every simulator with an :class:`AtomicSimEnv` gets Scheme D for free. Only
    the scripted demo itself (which privileged poses to servo to, in what order) is
    per-simulator.
    """

    def __init__(self, backend: AtomicSimEnv, max_cmd_m: float = SERVO_MAX_CMD_M) -> None:
        self.backend = backend
        self.max_cmd_m = float(max_cmd_m)
        self.grip_cmd = 1.0
        self.tcp: list[np.ndarray] = []
        self.grip_cmds: list[float] = []

    def _tcp(self) -> np.ndarray:
        return self.backend.tcp_pos()

    def capture(self) -> None:
        self.tcp.append(self._tcp())
        self.grip_cmds.append(self.grip_cmd)

    def step(self, delta_m: np.ndarray) -> None:
        self.backend.apply_delta(np.asarray(delta_m, dtype=np.float64), self.grip_cmd,
                                 self.max_cmd_m)
        self.capture()

    def servo_to(self, target, tol: float = 0.004, budget: int = 120) -> None:
        for _ in range(budget):
            err = np.asarray(target, dtype=np.float64) - self._tcp()
            if np.linalg.norm(err) < tol:
                return
            self.step(err)

    def set_gripper(self, close: bool, steps: int = 10) -> None:
        self.grip_cmd = -1.0 if close else 1.0
        for _ in range(steps):
            self.step(np.zeros(3))

    def hold_until_success(self, steps: int) -> bool:
        for _ in range(steps):
            self.step(np.zeros(3))
            if self.backend.success():
                return True
        return self.backend.success()


def gripper_events(grip_cmds: Sequence[float]) -> list[list]:
    """Index + token of every gripper transition in a recorded command track."""
    events = []
    prev = 1.0
    for i, g in enumerate(grip_cmds):
        if g < 0 <= prev:
            events.append([i, GRASP])
        elif g >= 0 > prev:
            events.append([i, RELEASE])
        prev = g
    return events


# ---------------------------------------------------------------------- writer
class RolloutWriter:
    """Teleop-format rollout writer: images + ``actions.jsonl`` + ``metadata.json``.

    The layout is EXACTLY what the real-robot teleop recorder produces
    (``agentview/NNNN.png`` + ``wrist/NNNN.png`` + ``actions.jsonl`` + ``metadata.json``),
    so ``train/data_preparation/rollouts_to_alpaca.py`` consumes sim rollouts
    unchanged and sim/real data mixes without a special case. Frame semantics match teleop:
    the stored frame is the observation BEFORE the token executes -- (obs_t, a_t) pairs --
    and ``gripper_closed`` is the gripper state BEFORE the token.

    Frames arrive ALREADY in their final form: the backend applies the whole camera
    transform contract to both views (``prepared_pair``), so this class only writes what
    it is given. It used to letterbox the agentview itself -- one view, in the writer,
    while the crop and flip lived in the backend -- and the wrist was documented as
    "already square and never touched", which is true of the simulators and false of every
    real rig this data is meant to match.

    ``agentview_square`` is accepted and ignored, so old call sites keep working.
    """

    def __init__(self, rollout_dir: Path, agentview_square: Optional[int] = None) -> None:
        self.dir = Path(rollout_dir)
        (self.dir / "agentview").mkdir(parents=True, exist_ok=True)
        (self.dir / "wrist").mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.dir / "actions.jsonl").open("w", encoding="utf-8")
        self.step = 0
        self.tokens: list[str] = []
        # Kept for call-site compatibility only; the backend owns the letterbox now.
        self.agentview_square = None

    def add_step(
        self,
        token: str,
        kind: str,
        agentview: np.ndarray,
        wrist: np.ndarray,
        gripper_closed: bool,
        ee_pose: list[float],
        width: float,
    ) -> None:
        from PIL import Image

        if self.agentview_square:
            from core.franka.camera_utils import resize_with_pad

            agentview = resize_with_pad(
                agentview, self.agentview_square, self.agentview_square
            )

        name = f"{self.step:04d}.png"
        Image.fromarray(agentview).save(self.dir / "agentview" / name)
        Image.fromarray(wrist).save(self.dir / "wrist" / name)
        rec = {
            "step": self.step,
            "token": token,
            "kind": kind,
            "gripper_closed": bool(gripper_closed),
            "ee_pose": ee_pose,
            "gripper_width": round(float(width), 5),
            "agentview": f"agentview/{name}",
            "wrist": f"wrist/{name}",
            "time": round(time.time(), 3),
        }
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()
        self.tokens.append(token)
        self.step += 1

    def close(self, metadata: dict) -> None:
        self._jsonl.close()
        counts: dict[str, int] = {}
        for t in self.tokens:
            counts[t] = counts.get(t, 0) + 1
        metadata = dict(metadata)
        metadata.update({"num_steps": self.step, "token_counts": counts})
        (self.dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )


# --------------------------------------------------------------------- episode
class TokenEpisode:
    """Record-then-execute loop + the token planners every generator shares.

    A generator (privileged oracle, demo follower, ...) supplies only the task logic and
    calls :meth:`emit` / :meth:`align_xy` / :meth:`go_z` / :meth:`chase`; this class keeps
    the frame/label contract (frame BEFORE the token) and the lattice guards in one place.

    All tolerances are clamped to >= 0.55 * ``step_m``: anything tighter than half a step
    makes ping-pong around the target geometrically inevitable, because a single 2 cm token
    jumps across the whole tolerance band.
    """

    def __init__(
        self,
        backend: AtomicSimEnv,
        writer: RolloutWriter,
        executor: AtomicExec,
        max_tokens: int = 160,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.backend = backend
        self.writer = writer
        self.exec = executor
        self.max_tokens = int(max_tokens)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    # -- recorded execution --------------------------------------------------
    def emit(self, token: str, kind: Optional[str] = None) -> None:
        """Record (frame BEFORE action, token), then execute the token."""
        if self.writer.step >= self.max_tokens:
            raise TokenBudgetExceeded("token budget exceeded")
        if self.backend.frozen():
            # Stop BEFORE recording: the frame would be identical to the previous one and
            # the label would claim a move that cannot happen (see AtomicSimEnv.frozen).
            raise EpisodeComplete("environment terminated; no further tokens are real")
        kind = kind or token_kind(token)
        agentview, wrist = self.backend.grab_frames()
        self.writer.add_step(
            token=token,
            kind=kind,
            agentview=agentview,
            wrist=wrist,
            gripper_closed=self.exec.gripper_closed,
            ee_pose=self.backend.tcp_pose7(),
            width=self.backend.gripper_width(),
        )
        if kind == "move":
            self.exec.move(token)
        elif token == GRASP:
            self.exec.grasp()
        elif token == RELEASE:
            self.exec.release()

    def run_tokens(self, tokens: Sequence[str]) -> None:
        for t in tokens:
            self.emit(t, "move")

    # -- planners ------------------------------------------------------------
    def _tol(self, tol: float) -> float:
        return max(float(tol), self.exec.step_m * 0.55)

    def align_xy(self, target: np.ndarray, tol: float = 0.011,
                 attempts: int = 4) -> None:
        """Manhattan-align XY, re-planning between axis runs (closed loop on drift).

        The axis order is coin-flipped per attempt so the dataset does not always show
        "X first, then Y" -- teleop data has both.
        """
        tol = self._tol(tol)
        target_xy = np.asarray(target, dtype=np.float64).reshape(-1)[:2]
        for _attempt in range(attempts):
            err = target_xy - self.backend.tcp_pos()[:2]
            if np.all(np.abs(err) < tol):
                return
            axis_order = [0, 1] if self.rng.random() < 0.5 else [1, 0]
            toks = manhattan_tokens(np.array([err[0], err[1], 0.0]),
                                    step_m=self.exec.step_m, min_residual_m=tol,
                                    axis_order=axis_order + [2])
            if not toks:
                return
            self.run_tokens(toks)

    def go_z(self, target_z: float, tol: float = 0.011, attempts: int = 3) -> None:
        tol = self._tol(tol)
        for _attempt in range(attempts):
            dz = float(target_z) - self.backend.tcp_pos()[2]
            if abs(dz) < tol:
                return
            toks = manhattan_tokens(np.array([0.0, 0.0, dz]),
                                    step_m=self.exec.step_m, min_residual_m=tol)
            if not toks:
                return
            self.run_tokens(toks)

    def chase(self, waypoint: np.ndarray, tol: float, budget: int = 60,
              stall_frac: float = 0.3, stall_limit: int = 3) -> None:
        """Dominant-axis pursuit with an axis commit, a lattice guard and a stall guard.

        The axis lock commits to one axis until it is done (teleop-like runs of a single
        key); an immediate opposite token means the lattice cannot get any closer --
        converged, stop, rather than oscillate forever.

        The STALL guard stops a waypoint that the arm physically cannot reach: joint
        limits, a workspace boundary or a collision leave the executor commanding a full
        token while the TCP barely moves. Without it the loop happily burns its whole
        budget emitting tokens that go nowhere, and every one of those is written to the
        dataset as a labelled move over an unchanged frame -- mislabelled data that the
        step-size statistics only show as a `tokens_blocked` count after the fact.
        (Observed on RoboLab: 60 consecutive MV_DOWN with z fixed at 0.475, half of one
        episode.) A token achieving less than ``stall_frac`` of ``step_m``, ``stall_limit``
        times in a row, ends the pursuit.
        """
        tol = self._tol(tol)
        prev_token: Optional[str] = None
        axis_lock: Optional[int] = None
        stalled = 0
        min_progress = stall_frac * self.exec.step_m
        for _ in range(budget):
            err = np.asarray(waypoint, dtype=np.float64) - self.backend.tcp_pos()
            if axis_lock is not None and abs(err[axis_lock]) >= tol:
                axis = axis_lock
            else:
                axis = int(np.argmax(np.abs(err)))
                if abs(err[axis]) < tol:
                    return
                axis_lock = axis
            token = token_for_axis(axis, err[axis])
            if prev_token is not None and opposite(token, prev_token):
                return
            before = self.backend.tcp_pos()
            self.emit(token, "move")
            if float(np.linalg.norm(self.backend.tcp_pos() - before)) < min_progress:
                stalled += 1
                if stalled >= stall_limit:
                    return
            else:
                stalled = 0
            prev_token = token

    def settle(self, n: int) -> bool:
        """Hold still up to ``n`` control steps, returning as soon as success fires.

        Static success predicates (object at rest on the target) need a few quiet steps
        after the last token before they can turn true.
        """
        return self.settle_steps(n) is not None

    def settle_steps(self, n: int) -> Optional[int]:
        """:meth:`settle`, but returning HOW MANY steps it took (``None`` = never fired).

        Worth measuring rather than assuming. A settle budget that is too small does not
        look like a failure: the caller concludes "not successful yet" and does something
        else -- and if that something else takes time (a corrective re-chase, say), the
        predicate fires during it and the wrong action gets the credit. On RoboLab that
        turned into ~3 spurious backward tokens at the end of every episode, emitted after
        the object was already released. Recording the real number in the episode metadata
        makes the margin visible instead of leaving it to be rediscovered.
        """
        for i in range(int(n)):
            self.exec.hold(1)
            if self.backend.success():
                return i + 1
        return None


class EpisodeComplete(RuntimeError):
    """The sim stopped responding -- the episode is over, not failed.

    Distinct from :class:`TokenBudgetExceeded`: that one means the planner ran away, this
    one means the task finished and any further token would be recorded over a frozen
    scene. Callers should treat it as a normal end and go straight to scoring.
    """


class TokenBudgetExceeded(RuntimeError):
    """Raised by :meth:`TokenEpisode.emit` when an episode runs past ``max_tokens``."""


# -------------------------------------------------------------------- planning
def manhattan_tokens(delta: np.ndarray, step_m: float = 0.02,
                     min_residual_m: float = 0.01,
                     axis_order: Optional[list[int]] = None) -> list[str]:
    """Decompose a displacement into a Manhattan token run.

    Whole axes are emitted in ``axis_order`` (default: largest |delta| first), each as
    round(|d|/step_m) repeats of one token; a trailing residual >= min_residual_m gets one
    extra token. Mirrors how teleop data looks: runs of the same key, axis by axis.
    """
    delta = np.asarray(delta, dtype=np.float64)
    order = axis_order if axis_order is not None else list(np.argsort(-np.abs(delta)))
    out: list[str] = []
    for axis in order:
        d = float(delta[axis])
        n = int(round(abs(d) / step_m))
        if n == 0 and abs(d) >= min_residual_m:
            n = 1
        out.extend([token_for_axis(axis, np.sign(d))] * n)
    return out


def rdp(points: np.ndarray, eps: float = 0.008) -> list[int]:
    """Ramer-Douglas-Peucker on a 3D polyline; returns the kept sample indices.

    Used by the demo follower to reduce a recorded continuous TCP path to its corner
    waypoints, which are then chased with single-axis ``step_m`` tokens.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return list(range(len(pts)))
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            d = np.linalg.norm(pts[a + 1:b] - pts[a], axis=1)
        else:
            t = np.clip(((pts[a + 1:b] - pts[a]) @ seg) / (seg_len**2), 0, 1)
            proj = pts[a] + t[:, None] * seg
            d = np.linalg.norm(pts[a + 1:b] - proj, axis=1)
        imax = int(np.argmax(d))
        if d[imax] > eps:
            m = a + 1 + imax
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return [int(i) for i in np.where(keep)[0]]
