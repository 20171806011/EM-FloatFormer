"use strict";
const gate=require("./gate.cjs"), synthetic=require("./synthetic.cjs");
function run() {
  let checks=0; const rejected=[];
  function check(ok,why) { if(!ok) throw Error(why); checks++; }
  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  const clone=x=>JSON.parse(JSON.stringify(x));
  function bad(label, fragment, action) {
    let got=""; try { action(); } catch(e) { got=e.message; }
    check(got.includes(fragment),label+": "+got); rejected.push(label);
  }
  const examples={A:[[0,0,10],[0,10,0],[1,0,10],[1,10,0]],B:[[0,0,2],[1,0,2],[0,1,3],[1,9,0]]};
  const enumeration={};
  for (const [name,rows] of Object.entries(examples)) {
    const sums=[0,0], gates=["00","01","10","11"];
    for (const [z,s,d] of rows) sums[z]+=s-d;
    const decisions=a=>a.map(x=>Number(x>0)).join("");
    const opt=decisions(sums), losses=Object.fromEntries(gates.map(g=>[g,rows.reduce((a,[z,s,d])=>a+(g[z]==="1"?d:s),0)]));
    const val=Math.min(losses["00"],losses["11"])-losses[opt];
    check(opt==="00"||opt==="01","unexpected optimum");
    check(2*val===Math.abs(sums[0])+Math.abs(sums[1])-Math.abs(sums[0]+sums[1]),"value");
    check((val>0)===(Math.min(...sums)<0 && Math.max(...sums)>0),"strict improvement");
    const oracle=rows.reduce((a,[,s,d])=>a+Math.min(s,d),0);
    for(const g of gates) check(losses[g]-losses[opt]===sums.reduce((a,x,z)=>a+(g[z]===opt[z]?0:Math.abs(x)),0),"regret");
    for(const a of [-20,0,5,20]) for(const b of [-20,0,5,20])
      check(losses[decisions([a,b])]-losses[opt]<=Math.abs(a-sums[0])+Math.abs(b-sums[1]),"plugin L1");
    check(same(gates.map(g=>losses[g]),name==="A"?[20,20,20,20]:[10,3,14,7]),"enumeration");
    check(oracle===(name==="A"?0:1),"oracle");
    enumeration[name]={bits:40,gates,errors:gates.map(g=>losses[g]),oracle,optimal:opt,gain:val};
  }
  const {plan,dev,test}=synthetic(), m=gate.fit(plan,dev), base=gate.score(plan,m,test);
  const raw=base.strata.C0_raw_ota, stress=base.strata.stress;
  check(m.threshold===0 && Math.abs(m.cells[3].eta-0.35)<1e-12,"dev lock");
  check(same([raw.eS,raw.eD,raw.eG,raw.uS,raw.hS,raw.uD,raw.hD],[0,4,2,0,2,2,0]),"raw transitions");
  check(same([stress.eS,stress.eD,stress.eG,stress.uS,stress.hS,stress.uD,stress.hD],[10,3,1,9,0,2,0]),"stress transitions");
  check(base.raw.harm===0.1 && base.inference.status==="DESCRIPTIVE_ONLY","NI guard");
  check(same([base.pooled.bits,base.pooled.netS,base.pooled.netD,base.pooled.uS,base.pooled.hS,base.pooled.uD,base.pooled.hD],[40,7,4,9,2,4,0]),"pooled pairs");
  check(base.raw.noninferiority==="DESCRIPTIVE_ONLY" && base.raw.upper===null,"raw verdict");
  check(same([base.primaryRows,base.uniqueHosts,base.sessions,base.blocks],[4,2,1,1]),"independent units");
  check(base.byFrame.length===4 && base.bySession.length===1 && base.byBlock.length===1,"ledgers");
  check(Math.abs(base.weighted.vsBest-0.1)<1e-14,"best fixed");
  const tight=synthetic(0), m0=gate.fit(tight.plan,tight.dev);
  check(m0.threshold===null && gate.score(tight.plan,m0,tight.test).raw.harm===0,"m=0 fallback");
  const tech=clone(test);
  tech.rows.push({...clone(test.rows[0]),rep:"secondary",D:"1111111111"});
  const r=gate.score(plan,m,tech);
  check(r.technicalExcluded===1 && r.uniqueHosts===2 && same(base.strata,r.strata) && same(base.inference,r.inference),"technical repeat");
  check(!gate.decision(m,{residual:null,disagreement:1}) && !gate.decision(m,{residual:2,disagreement:1}),"unknown fallback");
  check(gate.compare({truth:"01",S:"?1",D:"0?"},true).uS===1 && gate.compare({truth:"01",S:"?1",D:"0?"},true).hS===1,"erasure scoring");
  // The gate API only sees Z. Changing test labels preserves every decision.
  const flipped=clone(test); for (const x of flipped.rows) x.truth="1111111111";
  check(same(gate.score(plan,m,flipped).byFrame.map(r=>r.gate),base.byFrame.map(r=>r.gate)),"label isolation");
  bad("test into fit","outside authorized",()=>gate.fit(plan,{alg:dev.alg,rows:[...dev.rows,...test.rows]}));
  bad("dev into score","outside authorized",()=>gate.score(plan,m,{alg:test.alg,rows:[...test.rows,...dev.rows]}));
  bad("overlapping split","overlap",()=>{const p=clone(plan);p.ranges[2].lo=7;gate.planInfo(p);});
  bad("guard too short","guard",()=>{const p=clone(plan);p.guard=3;gate.planInfo(p);});
  bad("missing primary","missing primary",()=>gate.score(plan,m,{alg:test.alg,rows:test.rows.slice(1)}));
  bad("duplicate original","duplicate technical",()=>gate.score(plan,m,{alg:test.alg,rows:[...test.rows,test.rows[0]]}));
  bad("changed payload","payload changed",()=>{const d=clone(test);d.rows[1].truth="1111111111";gate.score(plan,m,d);});
  bad("label as feature","feature whitelist",()=>{const d=clone(test);d.rows[0].z.trueCfo=1;gate.score(plan,m,d);});
  bad("algorithm drift","algorithm mismatch",()=>{const d=clone(test);d.alg.static="d".repeat(64);gate.score(plan,m,d);});
  bad("reuse confirmation session","confirmation session reused",()=>{const p=clone(plan);p.design="prospective_independent";p.evidence="synthetic";gate.score(p,m,test);});
  const p=clone(plan);p.design="prospective_independent";p.evidence="Synthetic branch check only";
  p.ranges[2].session="new1";
  const t=clone(test);for(const row of t.rows) row.session="new1";
  check(gate.score(p,m,t).inference.status==="DESCRIPTIVE_ONLY","single attested session");
  p.ranges.push({...p.ranges[2],session:"new2"});
  t.rows.push(...t.rows.map(row=>({...row,session:"new2"})));
  check(gate.score(p,m,t).inference.status==="NOT_ESTABLISHED","small K");
  p.ranges[p.ranges.length-1].hi=10;t.rows=t.rows.filter(x=>!(x.session==="new2"&&x.tti===11));
  check(gate.score(p,m,t).inference.status==="DESCRIPTIVE_ONLY","unequal quotas");
  return {synthetic_only:true,checks,rejected,enumeration,staticMarginZero:{threshold:m0.threshold,open:0},
    illustrativeMarginPointOne:{static:{U:raw.uS+stress.uS,H:raw.hS+stress.hS,net:raw.netS+stress.netS},
    dynamic:{U:raw.uD+stress.uD,H:raw.hD+stress.hD,net:raw.netD+stress.netD},rawHarm:base.raw.harm,
    status:base.inference.status,uniqueHosts:base.uniqueHosts},extraTechnicalRowsExcluded:r.technicalExcluded};
}
module.exports=run;
if (require.main===module) console.log(JSON.stringify(run(),null,2));
