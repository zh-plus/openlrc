#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestLazyImports(unittest.TestCase):
    FORBIDDEN_ROOTS = {"faster_whisper", "spacy", "tiktoken", "torch", "lingua"}

    def _loaded_modules_after(self, statement: str):
        script = textwrap.dedent(
            f"""
            import json
            import sys

            {statement}

            interesting = [
                name
                for name in sys.modules
                if name == "openlrc.openlrc"
                or name.split(".")[0] in {sorted(self.FORBIDDEN_ROOTS)}
            ]
            print(json.dumps(sorted(interesting)))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        )
        return json.loads(result.stdout)

    def test_config_import_does_not_load_openlrc_or_heavy_dependencies(self):
        loaded = self._loaded_modules_after("from openlrc import TranscriptionConfig, TranslationConfig")
        self.assertNotIn("openlrc.openlrc", loaded)
        self.assertEqual([name for name in loaded if name.split(".")[0] in self.FORBIDDEN_ROOTS], [])

    def test_lrcer_import_does_not_load_heavy_runtime_dependencies(self):
        loaded = self._loaded_modules_after("from openlrc import LRCer")
        self.assertEqual([name for name in loaded if name.split(".")[0] in self.FORBIDDEN_ROOTS], [])
