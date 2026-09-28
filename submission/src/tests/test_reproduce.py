"""check complete-model requirements and hash-bound stage resumption"""

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location('reproduce', Path(__file__).resolve().parents[2] / 'src/reproduce.py')
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


def test_score_only_payload_is_not_a_complete_model_bundle(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': {}}))
    with pytest.raises(ValueError, match='incomplete trained model inventory'):
        reproduce.verify(tmp_path)


def test_stage_resume_rejects_changed_output(tmp_path):
    out = tmp_path / 'result.txt'
    cmd = [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("complete")', str(out)]
    env = dict(os.environ, AMAZITES_REPRO_BINDING='fixed')
    reproduce.stage('sample', cmd, tmp_path, env, [out])
    reproduce.stage('sample', cmd, tmp_path, env, [out])
    out.write_text('changed')
    with pytest.raises(ValueError, match='stage receipt differs'):
        reproduce.stage('sample', cmd, tmp_path, env, [out])
