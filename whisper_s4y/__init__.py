import logging
import os
import sys

log = logging.getLogger(__name__)
_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s"))
log.addHandler(_log_handler)
log.setLevel(logging.DEBUG)

_cwd = os.path.dirname(os.path.abspath(__file__))
_module_root = _cwd
_project_root = os.path.dirname(_module_root)

_artefacts_dir = os.path.join(_project_root, 'artefacts')


def artefact_path(artefact_name: str) -> str:
    p = os.path.join(_artefacts_dir, artefact_name)
    pp = os.path.dirname(p)
    if not os.path.exists(pp):
        os.makedirs(pp)
    return p