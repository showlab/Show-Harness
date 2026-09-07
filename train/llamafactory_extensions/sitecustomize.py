"""Install the mvtoken extensions at interpreter startup, if this dir is on PYTHONPATH.

Why sitecustomize: multi-GPU runs are re-launched by upstream's launcher through torchrun,
whose workers know nothing about what the parent imported. torchrun does pass the
environment through, so putting this directory on PYTHONPATH makes every worker load it
during startup.

Without PYTHONPATH this file is never seen and nothing is installed - the default path
stays pure upstream. train/scripts/train.sh sets it only when the extensions are needed.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_to_other_sitecustomize() -> None:
    """Also load any pre-existing sitecustomize.

    Python imports only the FIRST sitecustomize it finds, and we get there first via
    PYTHONPATH. If this venv already had one, we would silently shadow it - a hard bug to
    track down. So drop ourselves from sys.path, give the other one a chance, restore.
    """
    ours = sys.modules.pop("sitecustomize", None)
    removed = [p for p in sys.path if p and os.path.abspath(p) == _HERE]
    for p in removed:
        sys.path.remove(p)
    try:
        import sitecustomize  # noqa: F401  # the other one; importing is what applies it
    except ImportError:
        pass
    except Exception as e:  # their failure should not take training down
        print(f"[mvtoken_ext] chaining to the existing sitecustomize failed: {e}", file=sys.stderr)
    finally:
        sys.path[:0] = removed
        if ours is not None:
            sys.modules["sitecustomize"] = ours


_chain_to_other_sitecustomize()

try:
    import mvtoken_ext

    mvtoken_ext.apply()
except Exception as e:
    # Fail loudly: running on silently means training something other than intended.
    print(f"[mvtoken_ext] failed to load extensions: {e}", file=sys.stderr)
    raise
