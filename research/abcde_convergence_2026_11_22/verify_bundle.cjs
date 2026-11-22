'use strict';
// Node 18+. Verifies file integrity, replay-independent references, and the portable test suite.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const root = __dirname;
const read = name => fs.readFileSync(path.join(root, name));
const contents = name => JSON.parse(read(name).toString('utf8'));
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const run = (script, args=[]) => execFileSync(process.execPath, [path.join(root,script), ...args], {
  cwd:root, encoding:'utf8', maxBuffer:32*1024*1024
});
for (const line of read('SHA256SUMS').toString('utf8').trimEnd().split('\n')) {
  const match = /^([a-f0-9]{64})  ([^\n]+)$/.exec(line);
  assert.ok(match, 'invalid checksum line');
  assert.equal(sha256(read(match[2])), match[1], match[2]);
}
for (const [script,expected] of [['A/verify.cjs','A/reference.json'], ['B/mini.js','B/mini.json'],
    ['B/fullgrid.js','B/fullgrid.json'], ['D/enumerate.cjs','D/results.json']]) {
  function compare(actual, reference) {
    if (typeof reference === 'number') {
      assert.ok(Number.isFinite(actual) && Math.abs(actual-reference) <= 1e-9, script);
    } else if (reference && typeof reference === 'object') {
      assert.deepEqual(Object.keys(actual).sort(), Object.keys(reference).sort(), script);
      for (const key of Object.keys(reference)) compare(actual[key],reference[key]);
    } else {
      assert.equal(actual,reference,script);
    }
  }
  compare(JSON.parse(run(script)), contents(expected));
}
const a = JSON.parse(run('A/verify_ab.cjs'));
assert.equal(a.status,'PASS'); assert.equal(a.numeric_checks,196);
run('D/verify.cjs');
run('C/synthetic.cjs',['--check']);
assert.deepEqual(JSON.parse(run('E/selftest.cjs')), contents('E/expected/verification.json').engine);
const e = 'E/fixtures/';
const locked = sha256(read(e+'synthetic.lock.json'));
const score = JSON.parse(run('E/cli.cjs',['score',e+'plan.json',e+'synthetic.lock.json',locked,e+'test.json']));
assert.deepEqual(score,contents('E/expected/synthetic-score.json'));
const p='experiments/parity/';
const bitPositions=JSON.parse(run(p+'by_bit.cjs')).positions;
const expectedPositions=contents(p+'by_bit.json').positions;
assert.equal(bitPositions.length,48);
for(let i=0;i<bitPositions.length;i++){
  const a=bitPositions[i], b=expectedPositions[i];
  for(const k of Object.keys(b)) {
    if(typeof b[k]==='number') assert.ok(Math.abs(a[k]-b[k])<1e-9);
    else assert.equal(a[k],b[k]);
  }
}
const ab = JSON.parse(run(p+'oracle.cjs',[p+'fixture.json',p+'raw.json.gz.b64','A/reference.json','A/ab_fixture.json']));
assert.equal(ab.discrete.count,156); assert.equal(ab.qam.candidate_bits,6960); assert.equal(ab.shared_fixture.checks,152);
for (const group of Object.values(ab)) for (const error of Object.values(group.max_abs)) assert.ok(error<1e-11);
const whole='experiments/whole_frame_gate/';
const codec=require('./experiments/whole_frame_gate/vector_codec.cjs'), gate=require('./E/gate.cjs');
const dev=codec.decode(contents(whole+'dev_vectors.json')), test=codec.decode(contents(whole+'test_vectors.json'));
const plan=contents(whole+'plan.json'), lock=contents(whole+'lock.json'), archivedScore=contents(whole+'engine_score.json');
const fit=gate.fit(plan,dev);
assert.deepEqual(fit,lock.model);
assert.equal(sha256(read(whole+'lock.json')),archivedScore.audit.lock_sha256);
assert.equal(sha256(JSON.stringify(dev)+'\n'),archivedScore.audit.dev_sha256);
assert.equal(sha256(JSON.stringify(test)+'\n'),archivedScore.audit.test_sha256);
assert.deepEqual({audit:archivedScore.audit,...gate.score(plan,fit,test)},archivedScore);
console.log('PASS: SHA-256 manifest, A/B exact checks, full grid, C matched heads, D rotation, E gate and Python trace.');
