#!/usr/bin/env python3
"""Verify public text/binary source manifest and package public/P2 local weights."""
import hashlib,json,zipfile
from pathlib import Path

def archive(root=Path('.'),output=None,full=False):
    root=Path(root)
    manifest=json.loads((root/'MANIFEST_SHA256.json').read_text())
    paths=[root/x['file'] for x in manifest]+[root/'MANIFEST_SHA256.json']
    for x in manifest:
        p=root/x['file'];assert p.stat().st_size==x['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],p
    if full:
        for directory in ('p2_checkpoints','p2b_checkpoints','p2e_checkpoints'):
            paths+=sorted((root/directory).glob('*.pt'))
    if output is None:
        output=Path('EM-FloatFormer_P2_full.zip' if full else 'EM-FloatFormer_P2_public.zip')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,'EM-FloatFormer/'+p.relative_to(root).as_posix())
    print('Verified',len(manifest),'manifest entries; packaged',len(paths),'files in',output)
    return output

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--output')
    args=ap.parse_args();archive(Path('.'),Path(args.output) if args.output else None,args.full)
