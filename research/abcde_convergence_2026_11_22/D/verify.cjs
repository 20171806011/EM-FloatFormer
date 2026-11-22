// Independent receiver check. Run: node verify.cjs (reads sibling results.json).
// Only ideal, noiseless test points are constructed. No recorded IQ or fitted model is used.
'use strict';
const reference = require('./results.json');
const angles = [-135,-90,-45,45,90,135,180], onsets = [5,6,7,8];
const states = onsets.flatMap(s => angles.map(a => ({s,a})));
const points = Array.from({length:16}, (_,n) =>
  [(2*Math.floor(n/4)-3)/Math.sqrt(10),(2*(n%4)-3)/Math.sqrt(10)]);
const rows = Array.from({length:14}, (_,t)=>t);
const dataCounts = rows.map(t => t===2 ? 87 : 101);
function assert(ok,msg) { if (!ok) throw Error(msg); }
function rotate([x,y],deg) {
  const r = deg*Math.PI/180;
  return [x*Math.cos(r)-y*Math.sin(r), x*Math.sin(r)+y*Math.cos(r)];
}
function close(a,b) { return Math.hypot(a[0]-b[0],a[1]-b[1])<1e-8; }
function decode(v) { return points.findIndex(p=>close(p,v)); }
function phase(h,t) { return t>=h.s ? h.a : 0; }
function bits(n) { return [...n.toString(2).padStart(4,'0')].map(Number); }
function fraction(a,b) {
  let p=a, q=b;
  while(q) [p,q]=[q,p%q];
  return String(a/p)+'/'+String(b/p);
}
// For every relative rotation, membership of even ONE symbol agrees with that of all 16.
// This validates the one representative observation per row used for feasibility below.
for (const a of [0,...angles]) for (const b of [0,...angles]) {
  const hits=points.filter(p=>decode(rotate(rotate(p,a),-b))>=0).length;
  assert(hits===0 || hits===16,'partial overlap of rotated constellations');
}
const originalPilotTx = Array.from({length:14},(_,k)=>
  ({t:2,p:[k%4<2 ? -0.7:0.7,k%2===0 ? -0.7:0.7]}));
function observations(h,extra) {
  const probes=rows.map(t=>rotate(points[(7*t+3)%16],phase(h,t)));
  const tx=extra ? [...originalPilotTx,{t:8,p:[0.7,0.7]}] : originalPilotTx;
  return {probes,pilots:tx.map(({t,p})=>({t,p,y:rotate(p,phase(h,t))}))};
}
// The filter only receives observed complex samples, known pilot values, and optional S.
function possible({probes,pilots},givenS) {
  return states.filter(h=>(givenS===null || givenS===h.s) &&
    pilots.every(({t,p,y})=>close(rotate(p,phase(h,t)),y)) &&
    probes.every((y,t)=>decode(rotate(y,-phase(h,t)))>=0));
}
// Project the augmented observation back to row-2 pilots, then re-evaluate all candidates.
for (const truth of states) for (const givenS of [null,truth.s]) {
  const full=observations(truth,true);
  const removed={probes:full.probes,pilots:full.pilots.filter(p=>p.t===2)};
  const before=possible(observations(truth,false),givenS);
  const projected=possible(removed,givenS);
  assert(JSON.stringify(before)===JSON.stringify(projected),'pilot ablation altered candidates');
}
let checked=0;
for (const known of [false,true]) for (const extra of [false,true]) {
  const errors=[0,0,0,0];
  for (const truth of states) {
    const candidates=possible(observations(truth,extra),known ? truth.s : null);
    assert(candidates.length>0,'empty posterior');
    for (const t of rows) for (let n=0;n<16;n++) {
      const y=rotate(points[n],phase(truth,t)), votes=[0,0,0,0];
      for (const h of candidates) {
        const m=decode(rotate(y,-phase(h,t)));
        assert(m>=0,'invalid inverse symbol');
        bits(m).forEach((b,j)=>{votes[j]+=b;});
      }
      bits(n).forEach((bit,j)=>{
        const decision=Number(2*votes[j]>candidates.length); // ties choose 0
        errors[j]+=dataCounts[t]*Number(decision!==bit);
      });
    }
  }
  const denominator=16*states.length*dataCounts.reduce((a,b)=>a+b,0);
  const key=[extra?'original_plus_post':'original',known?'known':'hidden'];
  const match=reference.conditions.filter(r=>r.pilots===key[0] && r.onset===key[1]);
  assert(match.length===1,'missing or duplicated condition '+key);
  const r=match[0];
  assert(r.per_bit_state_position_count===denominator,'denominator '+key);
  assert(errors.every((e,j)=>e===r.per_bit_error_counts[j] &&
    fraction(e,denominator)===r.per_bit_ber[j]),'bit risk '+key);
  assert(fraction(errors.reduce((a,b)=>a+b),4*denominator)===r.frame_ber,'frame risk '+key);
  checked++;
}
assert(reference.conditions.length===checked,'unexpected conditions');
console.log('PASS: 64 rotation-pair support checks; 56 pilot removals; four physical-observation cases; all 16 bit risks.');
