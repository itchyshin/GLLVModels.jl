"""Process-level controls for P1 oracle verification before package loading."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / 'tools' / 'core070_postfit_policy_batch.R'

@unittest.skipUnless(shutil.which('Rscript'), 'Rscript required')
class OraclePreloadTests(unittest.TestCase):
    def invoke(self, cwd, library, destination):
        return subprocess.run(['Rscript', '--vanilla', str(RUNNER), str(library), str(destination)],
                              cwd=cwd, env=dict(os.environ, GLLVM_PARITY_PIN='P1'),
                              capture_output=True, text=True, timeout=30)

    def test_bad_build_stops_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); library = base / 'oracle' / 'library'; library.mkdir(parents=True)
            out = base / 'out'; result = self.invoke(ROOT, library, out)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((out / 'oracle-verify-preload.log').is_file())
            self.assertIn('verification failed before library load', result.stderr)
            self.assertNotIn('there is no package', result.stderr)

    def test_verified_directory_cannot_fall_back_to_other_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base / 'tools').mkdir()
            (base / 'tools' / 'core070_build_oracle.py').write_text('print("CORE070_ORACLE_VERIFY_PASS")\n')
            library = base / 'oracle' / 'library'; library.mkdir(parents=True)
            out = base / 'out'; result = self.invoke(base, library, out)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((out / 'oracle-verify-preload.log').read_text().strip(), 'CORE070_ORACLE_VERIFY_PASS')
            self.assertIn('there is no package', result.stderr)

if __name__ == '__main__':
    unittest.main()
