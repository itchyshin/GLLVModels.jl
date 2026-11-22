"""Synthetic structural controls; these do not qualify or admit a parity row."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import subprocess
import tempfile
import core070_postcoef_transition as guard
import core070_postfit_p1_receipts as receipts

ROOT=Path(__file__).resolve().parent.parent

HISTORICAL_ROW_SHA256 = "0ada7a1400b5a87eb4756444aba581b4fde4fce797165a6fa5f4ab13c19dcf94"
HISTORICAL_ROW_PATH = ROOT / "test/parity/fixtures/postcoef_old_target_row.json"

def load_historical_row(path=HISTORICAL_ROW_PATH):
    row = json.loads(path.read_bytes(), object_pairs_hook=guard._unique_object, parse_constant=guard._reject_constant)
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    if row.get("source_id") != guard.TARGET or digest != HISTORICAL_ROW_SHA256 or digest != guard.OLD_TARGET_SHA256:
        raise ValueError("historical POST-COEF target fixture differs")
    return row

class HistoricalFixtureControls(unittest.TestCase):
    def test_fixture_matches_recorded_guard_digest(self):
        row = load_historical_row()
        digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        self.assertEqual(digest, "0ada7a1400b5a87eb4756444aba581b4fde4fce797165a6fa5f4ab13c19dcf94")
        self.assertEqual(digest, guard.OLD_TARGET_SHA256)

    def test_fixture_matches_historical_source_when_available(self):
        import shutil
        git = shutil.which("git")
        if not git:
            self.skipTest("git executable unavailable")
        prefix = [git, "--no-replace-objects", "-C", str(ROOT)]
        env = receipts.postcoef_git_env()
        repo = subprocess.run(prefix + ["rev-parse", "--git-dir"], capture_output=True, env=env)
        if repo.returncode:
            self.skipTest("Git repository unavailable")
        source_id = "179daafbbc0ce67842352fb0e6fc53f0d5bf0f70"
        source_commit = subprocess.run(prefix + ["cat-file", "-e", source_id + "^{commit}"], capture_output=True, env=env)
        if source_commit.returncode:
            self.skipTest("historical source commit unavailable")
        result = subprocess.run(prefix + ["show", source_id + ":" + guard.MAP], capture_output=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        source = json.loads(result.stdout, object_pairs_hook=guard._unique_object,
                            parse_constant=guard._reject_constant)
        matches = [row for row in source["rows"] if row.get("source_id") == guard.TARGET]
        self.assertEqual(len(matches), 1)
        canonical = json.dumps(matches[0], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), HISTORICAL_ROW_SHA256)

    def test_corrupt_fixture_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.json"
            row = load_historical_row()
            row["note"] = "tampered"
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                load_historical_row(path)

class TransitionControls(unittest.TestCase):
    def setUp(self):
        row=load_historical_row()
        old={'rows':[row], 'counts':{'numeric_pass':0,'numeric_non_discriminating':1}}
        new=copy.deepcopy(old)
        newrow=new['rows'][0]
        newrow.update(evidence_tier='numeric',measured_against=guard.PIN,
                      evidence={'receipt':[guard.NEW],'tier':guard.TIER},
                      measured_result={'case_verdicts':{guard.CID:'PASS'},'batch_verifier':{guard.CID:'PASS'},'row_verdict':'PASS'},
                      note=guard.NOTE)
        new['counts']={'numeric_pass':1,'numeric_non_discriminating':0}
        self.old=json.dumps(old).encode(); self.new=json.dumps(new).encode()
        def board(raw,status,receipt,tier):
            digest=hashlib.sha256(raw).hexdigest()
            return (f'Inputs:\n\n- `{guard.MAP}` (1 rows, sha256 `{digest}`)\n\n'
                    f'## Totals by family\n\n| `family` | {status} | total |\n|---|---|---|\n'
                    f'| `postfit` | 1 | 1 |\n| `all` | 1 | 1 |\n\n'
                    f'## P1 twin fixtures present at this head\n\n## Rows\n\n'
                    f'| Row id | classification | status | evidence | detail |\n|---|---|---|---|---|\n'
                    f'| {guard.BOARD_ID} `{guard.TARGET}` | required_core; cases: {guard.CID} | {status} | {receipt} | family postfit; evidence_tier {tier}; measured_against {guard.PIN} |\n').encode()
        self.before=board(self.old,'NON-DISCRIMINATING','not bound; cited: non-binding '+guard.OLD,'numeric_non_discriminating')
        self.after=board(self.new,'EVIDENCED',guard.NEW,'numeric')
        assembled={'inputs':[{'path':guard.MAP,'sha256':hashlib.sha256(self.old).hexdigest(),'rows':1}], 'rows':[row]}
        self.assembled_old=(json.dumps(assembled,indent=1)+'\n').encode()
        assembled['rows']=[newrow];assembled['inputs'][0]['sha256']=hashlib.sha256(self.new).hexdigest()
        self.assembled_new=(json.dumps(assembled,indent=1)+'\n').encode()

    def test_three_file_transition(self):
        self.assertTrue(guard.structural_transition(self.before,self.after,self.old,self.new))
        self.assertTrue(guard.assembled_transition(self.assembled_old,self.assembled_new,self.old,self.new))

    def test_tampering_is_rejected(self):
        self.assertFalse(guard.structural_transition(self.before,self.after.replace(b'EVIDENCED',b'FAIL',1),self.old,self.new))
        doc=json.loads(self.assembled_new);doc['rows'][0]['note']='forged'
        self.assertFalse(guard.assembled_transition(self.assembled_old,(json.dumps(doc,indent=1)+'\n').encode(),self.old,self.new))
        self.assertFalse(guard.assembled_transition(self.assembled_old,json.dumps(json.loads(self.assembled_new)).encode(),self.old,self.new))

    def test_prior_signed_note_and_boolean_count_are_rejected(self):
        old=json.loads(self.old);old['rows'][0]['note']='Maintainer signed failure: FAIL; do not overwrite'
        altered=json.dumps(old).encode()
        board=self.before.replace(hashlib.sha256(self.old).hexdigest().encode(),hashlib.sha256(altered).hexdigest().encode())
        self.assertFalse(guard.structural_transition(board,self.after,altered,self.new))
        new=json.loads(self.new);new['counts']['numeric_non_discriminating']=False
        altered=json.dumps(new).encode()
        board=self.after.replace(hashlib.sha256(self.new).hexdigest().encode(),hashlib.sha256(altered).hexdigest().encode())
        self.assertFalse(guard.structural_transition(self.before,board,self.old,altered))

    def test_assembled_digest_and_unrelated_metadata_are_protected(self):
        for field in ('sha256','unexpected'):
            doc=json.loads(self.assembled_new)
            if field=='sha256': doc['inputs'][0]['sha256']='0'*64
            else: doc['unexpected']='forged'
            with self.subTest(field=field):
                self.assertFalse(guard.assembled_transition(self.assembled_old,(json.dumps(doc,indent=1)+'\n').encode(),self.old,self.new))

class HistoryControls(unittest.TestCase):
    BOARD='docs/dev-log/core070/true-parity-latest/scoreboard.md'
    ASSEMBLED='docs/dev-log/core070/true-parity-latest/case-map-assembled.json'

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.git('init','-q')
        self.fixture=TransitionControls('test_three_file_transition');self.fixture.setUp()
        for path,data in ((guard.MAP,self.fixture.old),(self.BOARD,self.fixture.before),(self.ASSEMBLED,self.fixture.assembled_old)):
            self.write(path,data)
        self.write('src/fixture.jl',b'# source\n')
        self.git('add','--',guard.MAP,self.BOARD,self.ASSEMBLED,'src/fixture.jl')
        self.commit('baseline');self.source=self.git('rev-parse','HEAD')

    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.DEVNULL,text=True).strip()
    def commit(self,message):
        self.git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm',message)
    def write(self,path,data):
        dest=self.root/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    def transition(self,include_assembled=True):
        items=[(guard.MAP,self.fixture.new),(self.BOARD,self.fixture.after)]
        if include_assembled: items.append((self.ASSEMBLED,self.fixture.assembled_new))
        for path,data in items:self.write(path,data)
        self.git('add','--',*[x[0] for x in items]);self.commit('synthetic transition')
    def accepts(self):return receipts.postcoef_ref_compatible(self.root,self.source,self.git('rev-parse','HEAD'))

    def test_synchronized_transition_and_source_rejection(self):
        self.transition();self.assertTrue(self.accepts())
        self.write('src/fixture.jl',b'# changed\n');self.git('add','src/fixture.jl');self.commit('source change')
        self.assertFalse(self.accepts())

    def test_missing_or_stale_assembled_map_rejected(self):
        self.transition(False);self.assertFalse(self.accepts())
        self.git('reset','--hard',self.source);self.transition()
        self.write(self.ASSEMBLED,self.fixture.assembled_old);self.git('add',self.ASSEMBLED);self.commit('stale map')
        self.assertFalse(self.accepts())

    def test_intermediate_signed_notes_cannot_be_erased(self):
        for path in (guard.MAP,self.ASSEMBLED):
            with self.subTest(path=path):
                self.git('reset','--hard',self.source)
                baseline=(self.root/path).read_bytes();doc=json.loads(baseline)
                next(row for row in doc['rows'] if row['source_id']==guard.TARGET)['note']='Maintainer signed failure: FAIL; do not overwrite'
                self.write(path,json.dumps(doc).encode());self.git('add',path);self.commit('signed note')
                self.write(path,baseline);self.git('add',path);self.commit('erase note')
                self.transition();self.assertFalse(self.accepts())

if __name__=='__main__': unittest.main()
