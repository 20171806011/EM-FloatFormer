"use strict";
// CommonJS; no packages. All BERs are fractions, not percentages.
const need = (ok, why) => { if (!ok) throw new Error(why); };
const phases = ["fit", "select", "test"];
const key = (...parts) => JSON.stringify(parts);
const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const label = x => typeof x === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(x) && !["constructor","prototype","toString","valueOf"].includes(x);
const whole = x => Number.isSafeInteger(x) && x >= 0;
const real = x => typeof x === "number" && Number.isFinite(x);
const keys = (x, names, why) => need(x && Object.keys(x).length===names.length && names.every(k=>own(x,k)), why);
function planInfo(p) {
  need(p.version === 1 && p.raw === "C0_raw_ota", "schema/raw");
  need(whole(p.bits) && p.bits>0 && whole(p.block) && p.block>0 && whole(p.guard), "bits/block/guard");
  need(real(p.margin) && p.margin>=0 && p.margin<=1 && typeof p.margin_basis==="string" && p.margin_basis.trim(), "margin/basis");
  need(whole(p.block_origin),"block origin");
  keys(p.alg, ["frontend","static","dynamic"], "algorithm ids");
  for (const h of Object.values(p.alg)) need(typeof h==="string" && /^[a-f0-9]{64}$/.test(h), "algorithm sha256");
  const cs=Object.keys(p.conditions);
  need(cs.length && cs.includes(p.raw) && cs.every(label), "conditions");
  let sum=0;
  for (const c of cs) {
    const q=p.conditions[c]; need(real(q.weight) && q.weight>0 && label(q.primary), "weight/replica"); sum+=q.weight;
  }
  need(Math.abs(sum-1)<1e-12, "weights must sum to one");
  need(["historical","prospective_independent"].includes(p.design) && typeof p.evidence==="string", "design");
  if (p.design==="prospective_independent") need(p.evidence.trim().length>0, "sampling evidence required");
  const ranges=new Map(), seen=new Set();
  for (const x of p.ranges) {
    need(label(x.session) && phases.includes(x.phase) && whole(x.lo) && whole(x.hi) && x.lo<=x.hi, "range");
    if (!ranges.has(x.session)) ranges.set(x.session,[]);
    ranges.get(x.session).push(x); seen.add(x.phase);
  }
  need(phases.every(s=>seen.has(s)), "missing split");
  for (const rs of ranges.values()) {
    rs.sort((a,b)=>a.lo-b.lo);
    for (let i=1;i<rs.length;i++) {
      const a=rs[i-1], b=rs[i];
      need(b.lo>a.hi, "overlap");
      if (a.phase!==b.phase) need(phases.indexOf(a.phase)<phases.indexOf(b.phase) && b.lo-a.hi-1>=p.guard, "chronology/guard");
    }
    if (p.design==="prospective_independent" && rs.some(x=>x.phase==="test"))
      need(rs.every(x=>x.phase==="test"), "confirmation session reused");
  }
  return {cs,ranges};
}
function readRows(p, data, allowed) {
  const {cs,ranges}=planInfo(p);
  keys(data, ["alg","rows"], "input header"); keys(data.alg,Object.keys(p.alg),"input alg fields");
  need(Object.keys(p.alg).every(k=>p.alg[k]===data.alg[k]), "algorithm mismatch");
  need(Array.isArray(data.rows),"rows");
  const primary=new Map(), seen=new Set(), truths=new Map(), rows=[];
  for (const r of data.rows) {
    keys(r,["session","tti","condition","rep","z","truth","S","D"],"row fields");
    need(ranges.has(r.session) && whole(r.tti) && cs.includes(r.condition) && label(r.rep),"row id");
    const range=ranges.get(r.session).find(x=>r.tti>=x.lo && r.tti<=x.hi);
    need(range && allowed.includes(range.phase),"row outside authorized split");
    keys(r.z,["residual","disagreement"],"feature whitelist");
    need(Object.values(r.z).every(v=>v===null || (real(v) && v>=0)),"invalid feature");
    need(typeof r.truth==="string" && r.truth.length===p.bits && /^[01]+$/.test(r.truth),"truth bits");
    for (const y of [r.S,r.D]) need(typeof y==="string" && y.length===p.bits && /^[01?]+$/.test(y),"prediction bits");
    const host=key(r.session,r.tti), id=key(r.session,r.tti,r.condition,r.rep);
    need(!seen.has(id), "duplicate technical row"); seen.add(id);
    need(!truths.has(host) || truths.get(host)===r.truth,"payload changed within host"); truths.set(host,r.truth);
    if (r.rep!==p.conditions[r.condition].primary) continue;
    const x={...r,phase:range.phase,block:Math.floor((r.tti-p.block_origin)/p.block)};
    primary.set(key(r.session,r.tti,r.condition),x); rows.push(x);
  }
  for (const rs of ranges.values()) for (const r of rs) if (allowed.includes(r.phase))
    for (let t=r.lo;t<=r.hi;t++) for (const c of cs)
      need(primary.has(key(r.session,t,c)), "missing primary host/condition");
  return {rows, rawRows:data.rows.length, technicalExcluded:data.rows.length-rows.length};
}
function compare(r, open) {
  const G=open?r.D:r.S, n=r.truth.length;
  const c={frames:1,opened:Number(open),bits:n,eS:0,eD:0,eG:0,uS:0,hS:0,uD:0,hD:0};
  for (let i=0;i<n;i++) {
    const s=r.S[i]!==r.truth[i], d=r.D[i]!==r.truth[i], g=G[i]!==r.truth[i];
    c.eS+=s; c.eD+=d; c.eG+=g;
    c.uS+=s&&!g; c.hS+=!s&&g; c.uD+=d&&!g; c.hD+=!d&&g;
  }
  return c;
}
const aggregate = (rows, decide) => {
  const v={frames:0,opened:0,bits:0,eS:0,eD:0,eG:0,uS:0,hS:0,uD:0,hD:0};
  for (const r of rows) { const c=compare(r,decide(r.z)); for (const k of Object.keys(v)) v[k]+=c[k]; }
  need(v.bits>0,"empty aggregate");
  v.netS=v.uS-v.hS; v.netD=v.uD-v.hD;
  need(v.netS===v.eS-v.eG && v.netD===v.eD-v.eG,"paired identity");
  return {...v,berS:v.eS/v.bits,berD:v.eD/v.bits,berG:v.eG/v.bits,dS:v.netS/v.bits,dD:v.netD/v.bits};
};
function summary(p, rows, decide) {
  const strata=Object.fromEntries(Object.keys(p.conditions).map(c=>[c,aggregate(rows.filter(r=>r.condition===c),decide)]));
  const avg=k=>Object.entries(strata).reduce((v,[c,t])=>v+p.conditions[c].weight*t[k],0);
  const dS=avg("dS"), dD=avg("dD");
  return {strata,weighted:{berS:avg("berS"),berD:avg("berD"),berG:avg("berG"),dS,dD,vsBest:Math.min(dS,dD)},rawHarm:-strata[p.raw].dS};
}
const names=["residual","disagreement"];
function locate(z,m) {
  if (!m.axis || names.some(k=>!real(z[k]) || z[k]<m.axis[k].lo || z[k]>m.axis[k].hi)) return -1;
  return 2*Number(z.residual>m.axis.residual.mid)+Number(z.disagreement>m.axis.disagreement.mid);
}
function decision(m,z) {
  const j=locate(z,m);
  return m.threshold!==null && j>=0 && m.cells[j].enabled && m.cells[j].eta>m.threshold;
}
function fit(p, data) {
  const {rows}=readRows(p,data,["fit","select"]);
  const fitRows=rows.filter(x=>x.phase==="fit"), select=rows.filter(x=>x.phase==="select");
  const clean=fitRows.filter(r=>names.every(k=>real(r.z[k])));
  const m={version:1,axis:null,cells:[],threshold:null};
  if (clean.length) m.axis=Object.fromEntries(names.map(k=>{
    const a=clean.map(r=>r.z[k]).sort((u,v)=>u-v), i=Math.floor((a.length-1)/2);
    return [k,{lo:a[0],mid:a[i]/2+a[Math.floor(a.length/2)]/2,hi:a[a.length-1]}];
  }));
  const tot=Object.fromEntries(Object.keys(p.conditions).map(c=>[c,p.bits*fitRows.filter(r=>r.condition===c).length]));
  for (let j=0;j<4;j++) {
    const group=fitRows.filter(r=>locate(r.z,m)===j);
    const support=new Set(group.filter(r=>r.condition===p.raw).map(r=>key(r.session,r.block)));
    let gain=0, bits=0;
    for (const r of group) {
      const q=p.conditions[r.condition].weight/tot[r.condition], a=compare(r,false);
      gain+=q*(a.eS-a.eD); bits+=q*a.bits;
    }
    m.cells.push({eta:bits?gain/bits:null,enabled:support.size>=2,support:support.size});
  }
  const thresholds=new Set([0,...m.cells.filter(c=>c.enabled&&c.eta>=0).map(c=>c.eta)]);
  let best=0, tBest=null; // always-S attainable, preferred on ties
  for (const t of [...thresholds].sort((a,b)=>b-a)) {
    m.threshold=t;
    const s=summary(p,select,z=>decision(m,z));
    if (s.rawHarm<=p.margin && s.weighted.dS>best) { best=s.weighted.dS; tBest=t; }
  }
  m.threshold=tBest; return m;
}
function score(p,m,data) {
  const d=readRows(p,data,["test"]), rows=d.rows, decide=z=>decision(m,z);
  const sessions=[...new Set(rows.map(r=>r.session))].sort();
  const blocks=[...new Set(rows.map(r=>key(r.session,r.block)))].sort().map(k=>JSON.parse(k));
  const bySession=sessions.map(session=>({session,...summary(p,rows.filter(r=>r.session===session),decide)}));
  const byBlock=blocks.map(([session,block])=>({session,block,...summary(p,rows.filter(r=>r.session===session && r.block===block),decide)}));
  const overall=summary(p,rows,decide), K=sessions.length;
  const attested=p.design==="prospective_independent";
  const equal=new Set(bySession.map(x=>x.strata[p.raw].bits)).size===1;
  let inference={status:"DESCRIPTIVE_ONLY",reason:"independent prospective sessions or equal quotas unavailable",eligibleSessions:0};
  if (attested && equal && K>=2) {
    const a=Math.sqrt(2*Math.log(3/0.05)/K);
    const lowerS=overall.weighted.dS-a,lowerD=overall.weighted.dD-a,upperRaw=overall.rawHarm+a;
    inference={status:lowerS>0&&lowerD>0&&upperRaw<=p.margin?"PASS":"NOT_ESTABLISHED",eligibleSessions:K,lowerS,lowerD,upperRaw,halfWidth:a,assumption:p.evidence};
  }
  return {version:1,primaryRows:rows.length,totalRows:d.rawRows,technicalExcluded:d.technicalExcluded,
    uniqueHosts:new Set(rows.map(r=>key(r.session,r.tti))).size,sessions:K,blocks:blocks.length,
    ...overall,pooled:aggregate(rows,decide),raw:{margin:p.margin,pointWithin:overall.rawHarm<=p.margin,harm:overall.rawHarm,upper:inference.upperRaw??null,noninferiority:inference.upperRaw===undefined?"DESCRIPTIVE_ONLY":(inference.upperRaw<=p.margin?"SUPPORTED":"NOT_ESTABLISHED")},
    inference,bySession,byBlock,byFrame:rows.map(r=>({session:r.session,tti:r.tti,block:r.block,condition:r.condition,rep:r.rep,gate:decide(r.z)?"D":"S",...compare(r,decide(r.z))}))};
}
module.exports={planInfo,readRows,compare,decision,fit,score};
