"use strict";
// Deterministic scenario B. Dummy algorithm hashes identify synthetic data only.
function synthetic(margin=0.1) {
  const alg={frontend:"a".repeat(64),static:"b".repeat(64),dynamic:"c".repeat(64)};
  const p={version:1,alg,raw:"C0_raw_ota",bits:10,block:2,block_origin:0,guard:2,
    margin,margin_basis:"Synthetic test tolerance; never a deployment margin",design:"historical",evidence:"",
    conditions:{C0_raw_ota:{weight:0.5,primary:"r0"},stress:{weight:0.5,primary:"r0"}},
    ranges:[{session:"syn",lo:0,hi:3,phase:"fit"},{session:"syn",lo:6,hi:7,phase:"select"},{session:"syn",lo:10,hi:11,phase:"test"}]};
  const rows=[], bits=k=>"1".repeat(k)+"0".repeat(10-k);
  for (const q of p.ranges) for(let t=q.lo;t<=q.hi;t++) for(const c of Object.keys(p.conditions)) {
    const high=t%2, raw=c==="C0_raw_ota", s=raw?0:(high?9:1), d=raw?2:(high?0:3);
    rows.push({session:"syn",tti:t,condition:c,rep:"r0",z:{residual:high,disagreement:high},
      truth:"0000000000",S:bits(s),D:bits(d)});
  }
  return {plan:p,dev:{alg,rows:rows.filter(x=>x.tti<10)},test:{alg,rows:rows.filter(x=>x.tti>=10)}};
}
module.exports=synthetic;
