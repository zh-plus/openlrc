#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestLazyImports(unittest.TestCase):
    FORBIDDEN_ROOTS = {"faster_whisper", "tiktoken", "torch", "lingua"}
    PROBE_MARKER = "__OPENLRC_LAZY_IMPORTS__="

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
            print({self.PROBE_MARKER!r} + json.dumps(sorted(interesting)))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith(self.PROBE_MARKER):
                return json.loads(line[len(self.PROBE_MARKER) :])

        self.fail(
            f"lazy import probe did not emit a parseable result. stdout={result.stdout!r}, stderr={result.stderr!r}"
        )

    def test_config_import_does_not_load_openlrc_or_heavy_dependencies(self):
        loaded = self._loaded_modules_after("from openlrc import TranscriptionConfig, TranslationConfig")
        self.assertNotIn("openlrc.openlrc", loaded)
        self.assertEqual([name for name in loaded if name.split(".")[0] in self.FORBIDDEN_ROOTS], [])

    def test_lrcer_import_does_not_load_heavy_runtime_dependencies(self):
        loaded = self._loaded_modules_after("from openlrc import LRCer")
        self.assertEqual([name for name in loaded if name.split(".")[0] in self.FORBIDDEN_ROOTS], [])

    def test_translate_path_does_not_load_media_utils_or_heavy_deps(self):
        """Importing only the translation-path modules must not pull in media_utils or heavy deps."""
        # The translate path legitimately loads lingua (via validators.py) and
        # tiktoken (via utils.get_text_token_number).  Only check for the truly
        # heavy packages that would bloat a Nuitka binary.
        heavy_roots = {"faster_whisper", "torch"}
        loaded = self._loaded_modules_after(
            "from openlrc.agents import create_chatbot; "
            "from openlrc.context import TranslateInfo; "
            "from openlrc.opt import SubtitleOptimizer; "
            "from openlrc.subtitle import BilingualSubtitle, Subtitle; "
            "from openlrc.translate import LLMTranslator"
        )
        self.assertNotIn("openlrc.media_utils", loaded)
        self.assertNotIn("openlrc.openlrc", loaded)
        self.assertEqual([name for name in loaded if name.split(".")[0] in heavy_roots], [])
