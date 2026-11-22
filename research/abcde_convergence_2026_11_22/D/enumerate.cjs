// EM-FloatFormer D, commit 5f88a5be984493bcb6aa164d84253b6f29876eef
// Run: node enumerate.cjs > regenerated.json (standard JavaScript, no dependencies).
'use strict';
const angles = [-135,-90,-45,45,90,135,180], onsets = [5,6,7,8];
const levels = [-3,-1,1,3];
const roots = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]];
function observe(n, angle) {
  const k = (angle/45+8)%8, [c,s] = roots[k];
  const i = levels[n >> 2], q = levels[n & 3];
  const a = c*i-s*q, b = s*i+c*q;
  // Each coordinate of 2*sqrt(10)*Y is u+v*sqrt(2); key=(uI,vI,uQ,vQ).
  return (k%2 ? [0,a,0,b] : [2*a,0,2*b,0]).join(',');
}
function add(groups, key, n) {
  if (!groups.has(key)) groups.set(key,[]);
  groups.get(key).push(n);
}
function bitErrors(groups) {
  const errors = [0,0,0,0];
  for (const states of groups.values()) {
    for (let j=0; j<4; ++j) {
      const ones = states.reduce((t,n)=>t+((n>>(3-j))&1),0);
      errors[j] += Math.min(ones, states.length-ones);
    }
  }
  return errors;
}
function check(condition) { if (!condition) throw Error('enumeration mismatch'); }
function fraction(a,b) {
  const gcd = (x,y) => y ? gcd(y,x%y) : x;
  const d = gcd(a,b); return (a/d)+'/'+(b/d);
}
const output = {
  schema: 'em-floatformer-d/v1', commit: '5f88a5be984493bcb6aa164d84253b6f29876eef',
  model: 'ideal_noiseless_iid_uniform_uncoded_16qam_known_channel', angles_deg: angles,
  onsets_zero_based: onsets, angle_prior: '1/7', onset_prior: '1/4',
  pilot_row: 2, extra_pilot: {row:8, tx:[0.7,0.7], resource:'additional_observation'},
  bit_order:['I_msb','I_lsb','Q_msb','Q_lsb'], classes:[], by_onset:[], conditions:[]
};
const post = new Map();
for (const angle of angles) for (let n=0;n<16;++n) add(post, observe(n,angle), n);
check(post.size===32);
for (const m of [3,4]) {
  const subset = new Map([...post].filter(([,v])=>v.length===m));
  check(subset.size===16);
  const errors = bitErrors(subset);
  output.classes.push({angles_deg:angles.filter(a=>(a%90!==0)===(m===4)), observations:subset.size,
    hypotheses_per_observation:m, per_bit_error_counts:errors, states:16*m});
  check(errors.every(x=>x===(m===3 ? 16 : 32)));
}
// For the integer-angle class the unique bitwise majority equals the label of -Y.
for (const a of [-90,90,180]) {
  const mistakes = [0,0,0,0];
  for (let n=0;n<16;++n) {
    const key = observe(n,a);
    const opposite = Array.from({length:16},(_,m)=>m).find(m=>observe(m,180)===key);
    for (let j=0;j<4;++j) mistakes[j] += ((n^opposite)>>(3-j))&1;
  }
  check(mistakes.every(v=>v===(a===180?0:8)));
}
const perBit = bitErrors(post);
check(perBit.every(x=>x===48));
output.post_ber_known_onset = perBit.map(x=>fraction(x,16*angles.length));
const pilots = [13,21,29,37,45,53,61,69,77,85,93,101,109,114];
const rowN = Array.from({length:14},(_,r)=>Array.from({length:102},(_,i)=>13+i)
  .filter(k=>k!==64 && !(r===2 && pilots.includes(k))).length);
const total = rowN.reduce((a,b)=>a+b,0);
check(total===1400 && rowN[2]===87);
output.data_re_by_row=rowN;
output.data_re=total;
output.data_bits=4*total;
for (const s of onsets) {
  const n = rowN.slice(s).reduce((a,b)=>a+b,0);
  output.by_onset.push({onset:s, post_re:n, original_known_s_frame_ber:fraction(n*perBit[0],total*16*angles.length)});
}
// In noiseless data the entire grid identifies S for odd multiples of 45 degrees;
// for even multiples every possible S has identical grid likelihood.
for (const known of [true,false]) for (const extra of [false,true]) {
  const errors = [0,0,0,0];
  for (let row=0;row<14;++row) {
    const groups = new Map();
    for (const s of onsets) for (const a of angles) for (let n=0;n<16;++n) {
      const odd = a%90!==0, sObs = (known || odd) ? s : '?';
      // Index 10 has I=Q=+1/sqrt(10), a known positive rescaling of 0.7+0.7j.
      const pilotAngle = extra ? observe(10,a) : '?';
      add(groups,[odd,sObs,pilotAngle,observe(n,row>=s ? a : 0)].join('|'),n);
    }
    bitErrors(groups).forEach((e,j)=>{errors[j]+=rowN[row]*e;});
  }
  const denom = total*onsets.length*angles.length*16;
  const result = errors.map(e=>fraction(e,denom));
  const target = known ? (extra?'0/1':'909/3920') : (extra?'101/4900':'303/1225');
  check(result.every(x=>x===target));
  output.conditions.push({pilots:extra?'original_plus_post':'original',onset:known?'known':'hidden',
    per_bit_error_counts:errors, per_bit_state_position_count:denom, per_bit_ber:result,
    frame_ber:fraction(errors.reduce((a,b)=>a+b,0),4*denom)});
}
console.log(JSON.stringify(output,null,2));
