"""Apply the primary reconstruction audit unchanged to the independent repeat."""
import sys
import audit_rebuild as audit

if __name__ == '__main__':
    audit.SEED_START = 981120040000
    audit.SEEDS = list(range(audit.SEED_START, audit.SEED_START+128))
    if len(sys.argv)==1:
        sys.argv.extend(['--input','rebuild_replication.json','--out','rebuild_replication_audit.json'])
    audit.main()
