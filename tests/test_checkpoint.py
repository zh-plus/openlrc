#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openlrc.checkpoint import save_json_checkpoint


class TestAtomicCheckpoint(unittest.TestCase):
    def test_atomically_replaces_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compare.json"
            path.write_text('{"old": true}', encoding="utf-8")

            save_json_checkpoint(path, {"new": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": True})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_failed_replace_preserves_previous_checkpoint_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compare.json"
            path.write_text('{"old": true}', encoding="utf-8")

            with patch("openlrc.checkpoint.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    save_json_checkpoint(path, {"new": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
