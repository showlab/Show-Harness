"""Smooth-motion tool: ramp each move's setpoint instead of stepping it, and CHAIN
consecutive moves so the arm flows through them instead of stopping at every token.

Without this tool the controller jumps the setpoint straight to the new target and
re-commands it a few times: the arm springs toward it with a hard initial jerk and a
sudden stop. Enabled, the controller walks the setpoint from the start pose to the target
along an eased profile, commanding intermediate waypoints ``dt_s`` apart.

Two things this tool solves, both of which caused the "step-and-stop" vibration on the
stiff Piper (whose ``/puppet/pos_cmd`` is a point-to-point EndPoseCtrl MOVE P, quantized
to 1 mm):

1. **Velocity continuity (blending).** A plain min-jerk profile starts AND ends at zero
   velocity, so every atomic token decelerates the arm to a full stop and the next one
   accelerates it from rest -- the arm pulses once per token. :meth:`plan` instead takes
   boundary velocities ``v0``/``v1``: the controller ends a move at CRUISE speed when it
   knows another aligned move follows (a held teleop key, an action chunk) and starts the
   next one at that same speed. The concatenated setpoint path is then C1-continuous:
   accelerate once, cruise (the profile degenerates to a straight constant-velocity line
   when ``v0 == v1 == cruise``), decelerate once at the end of the run.

2. **Sub-quantum crawl.** Min-jerk's end slopes are near zero, so its first/last waypoints
   advance the setpoint by far LESS than the hardware's command resolution: the arm sits
   still for several commands, then jumps a whole quantum -- visible stutter at the edge of
   every move. ``min_waypoint_m`` caps the waypoint COUNT so the average step stays at or
   above that spacing (the total move duration is preserved -- only the command density
   drops). Set it to a small multiple of the hardware resolution (Piper: ~2 mm).

The tool owns no pose math: it yields eased interpolation fractions (0->1) and the
per-waypoint delay; the controller interpolates start->target with them. Disabled -> the
controller keeps its plain settle behaviour (byte-identical to no tool), and with
``blend`` off / no ``continuous`` hint the profile is exactly the classic min-jerk.
"""
from __future__ import annotations

from typing import List, Tuple


class SmoothPlugin:
    """Min-jerk / cruise interpolation schedule for one setpoint move."""

    def __init__(
        self,
        enabled: bool = False,
        substeps: int = 20,
        dt_s: float = 0.05,
        min_waypoint_m: float = 0.0,
        blend: bool = True,
        cruise: float = 1.0,
    ) -> None:
        self.enabled = bool(enabled)
        # substeps * dt_s is the move DURATION; it is preserved even when min_waypoint_m
        # thins the waypoints out (dt grows to compensate), so smoothing never changes
        # how fast the arm travels -- only how finely the path is commanded.
        self.substeps = max(1, int(substeps))
        self.dt_s = max(0.0, float(dt_s))
        # Minimum setpoint advance per waypoint (m). 0 -> no floor (legacy behaviour).
        self.min_waypoint_m = max(0.0, float(min_waypoint_m))
        # Chain aligned consecutive moves at cruise speed instead of stopping between them.
        self.blend = bool(blend)
        # Boundary "velocity" of a chained move, in fraction-of-the-move per unit of
        # normalized time. 1.0 == constant rate, which makes a chained middle move a
        # straight line (perfectly uniform command spacing).
        self.cruise = float(cruise)

    @property
    def duration_s(self) -> float:
        """Total time one move's ramp takes (independent of the waypoint count)."""
        return self.substeps * self.dt_s

    def waypoint_count(self, distance_m: float) -> int:
        """How many waypoints to emit for a move of ``distance_m``.

        ``substeps``, thinned so the average advance per waypoint is at least
        ``min_waypoint_m`` -- otherwise the end of a min-jerk ramp commands sub-quantum
        increments the hardware cannot act on.
        """
        n = self.substeps
        if self.min_waypoint_m > 0.0 and distance_m > 0.0:
            n = min(n, int(distance_m / self.min_waypoint_m))
        return max(2, n)

    def plan(
        self, distance_m: float = 0.0, v0: float = 0.0, v1: float = 0.0
    ) -> Tuple[List[float], float]:
        """Return ``(fractions, dt_s)`` for one move.

        ``fractions`` are eased samples in (0, 1] to interpolate start->target;
        ``dt_s`` is the delay between them (chosen so the move still takes
        :attr:`duration_s` regardless of how many waypoints survive the spacing floor).

        ``v0``/``v1`` are the boundary velocities (0 = start/end at rest;
        :attr:`cruise` = flow through at cruise speed). The quintic
        ``s(t) = v0*t + a3 t^3 + a4 t^4 + a5 t^5`` is the unique profile with
        ``s(0)=0, s(1)=1, s'(0)=v0, s'(1)=v1, s''(0)=s''(1)=0``; with ``v0=v1=0`` it
        reduces exactly to the classic min-jerk ``10t^3 - 15t^4 + 6t^5``.
        """
        n = self.waypoint_count(distance_m)
        dt = (self.duration_s / n) if n > 0 else 0.0

        a1 = float(v0)
        A = 1.0 - float(v0)
        B = float(v1) - float(v0)
        a3 = 10.0 * A - 4.0 * B
        a4 = -15.0 * A + 7.0 * B
        a5 = 6.0 * A - 3.0 * B

        out: List[float] = []
        prev = 0.0
        for i in range(1, n + 1):
            t = i / n
            s = a1 * t + a3 * t**3 + a4 * t**4 + a5 * t**5
            # Never let a profile walk the setpoint BACKWARD (a malformed v0/v1 pair
            # could): the arm must only ever advance toward the target.
            s = max(s, prev)
            prev = s
            out.append(s)
        out[-1] = 1.0  # land exactly on the target
        return out, dt

    def fractions(self) -> List[float]:
        """Back-compat: the plain min-jerk fractions (rest-to-rest, no spacing floor)."""
        return self.plan(0.0, 0.0, 0.0)[0]
