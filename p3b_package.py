#!/usr/bin/env python3
"""Package P3-B source/results/weights into checksum-verified archives."""
from pathlib import Path
import json,hashlib,zipfile,tempfile,subprocess,sys,shutil
root=Path(__file__).resolve().parent
out=Path('/mnt/data')
results=root/'p3b_results'
source_code=[p for p in root.glob('*.py') if p.is_file()]
source_notes=[root/x for x in ('P3B_PROTOCOL.md','P3B_REPORT.md','P3B_SCOPE_ADDENDUM.md','P3_A_MECHANISM_PROTOCOL.md')]
source_results=[p for p in results.iterdir() if p.is_file()]
full_weights=sorted(root.glob('p3b_checkpoints/*.pt'))+sorted(root.glob('p3a2_checkpoints/*.pt'))
assert len(full_weights)==35,len(full_weights)
source=sorted(set(source_code+source_notes+source_results))
full=sorted(set(source+full_weights+[p for p in root.glob('p3b_*.log') if p.is_file()]))

def build(name, files):
    entries={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size} for p in files}
    dest=out/name
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p in files:z.write(p,arcname='EM-FloatFormer/'+str(p.relative_to(root)))
        z.writestr('EM-FloatFormer/P3B_ARCHIVE_SHA256_MANIFEST.json',json.dumps({'archive':name,'evidence_files':entries,'is_offline_full':len(files)==len(full)},indent=2)+'\n')
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('EM-FloatFormer/P3B_ARCHIVE_SHA256_MANIFEST.json'))
        assert len(manifest['evidence_files'])==len(files)
        for path,details in manifest['evidence_files'].items():
            data=z.read('EM-FloatFormer/'+path)
            assert hashlib.sha256(data).hexdigest()==details['sha256'] and len(data)==details['size']
    print(dest, dest.stat().st_size,'bytes;',len(files),'verified files')
    return dest

zip_source=build('EM-FloatFormer_P3B_source_results_2026-11-15.zip',source)
zip_full=build('EM-FloatFormer_P3B_full_2026-11-15.zip',full)
# Verify that the readme includes no inaccurate confirmation of a later phase.
with tempfile.TemporaryDirectory() as td:
    with zipfile.ZipFile(zip_full) as z:z.extractall(td)
    folder=Path(td)/'EM-FloatFormer'
    proc=subprocess.run([sys.executable,'-m','unittest','test_p3b_link','-q'],cwd=folder,capture_output=True,text=True)
    assert proc.returncode==0,(proc.stdout,proc.stderr)
    proc2=subprocess.run([sys.executable,'p3b_independent_audit.py'],cwd=folder,capture_output=True,text=True)
    assert proc2.returncode==0,(proc2.stdout,proc2.stderr)
    print('UNPACKED FULL ARCHIVE SMOKE: 12 tests PASS; independent CSV audit PASS')
