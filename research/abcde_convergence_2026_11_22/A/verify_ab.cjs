// Run: node verify_ab.cjs   (Node 18+, no dependencies)
'use strict';
const fixture = require('./ab_fixture.json');
const cfg = fixture.input;
const [targetRow, targetTone] = cfg.target;
const total = xs => xs.reduce((a,b) => a+b,0);
const normalize = xs => xs.map(x => x/total(xs));
const weighted = (w,x) => total(w.map((p,j) => p*x[j]));
const hard = ps => ps.map(p => Number(p>0.5)).join('');
const ones = (index,bit) => (index >> (3-bit)) & 1;
const bits = [0,1,2,3];
const alphabet = cfg.pam_levels_unscaled.flatMap(i => cfg.pam_levels_unscaled.map(q =>
  [i/Math.sqrt(10),q/Math.sqrt(10)]));
let comparisons=0, worstError=0;
function close(actual,expected,label) {
  if(Array.isArray(expected)) {
    if(!Array.isArray(actual) || actual.length!==expected.length) throw Error(label+': shape');
    expected.forEach((x,k) => close(actual[k],x,label+'['+k+']'));
  } else if(typeof expected==='string') {
    if(actual!==expected) throw Error(label+': '+actual+' vs '+expected);
  } else {
    const err=Math.abs(actual-expected);
    if(!Number.isFinite(err) || err>2e-12) throw Error(label+': '+actual+' vs '+expected);
    worstError=Math.max(worstError,err); comparisons++;
  }
}
function varianceGrid(masked) {
  const lastRow=cfg.y.length-1, lastTone=cfg.y[0].length-1;
  return cfg.y.map((row,s)=>row.map((_,k)=>{
    let power=0,count=0;
    for(let ds=-1;ds<=1;ds++) for(let dk=-1;dk<=1;dk++) {
      const u=Math.min(lastRow,Math.max(0,s+ds)), v=Math.min(lastTone,Math.max(0,k+dk));
      if(masked && u===targetRow && v===targetTone) continue;
      const [re,im]=cfg.y[u][v];power+=re*re+im*im;count++;
    }
    const mean=power/count, floor=cfg.variance_floor;
    return mean/(1+floor)>1.75 ? Math.min(20,Math.max(floor,mean-1)) : floor;
  }));
}
function symbolMass(y,H,v,x) {
  const dr=y[0]-(H[0]*x[0]-H[1]*x[1]);
  const di=y[1]-(H[0]*x[1]+H[1]*x[0]);
  return cfg.uniform_symbol_prior*Math.exp(-(dr*dr+di*di)/v)/(Math.PI*v);
}
// Each row has three carriers. Sum over all 16^3 assignments before applying any temperature.
// Row independence factors the six-symbol sum exactly. No reuse of B's log-score or softmax.
function sumRow(masses,row) {
  let all=0,without=0;
  const bitMass=bits.map(()=>0);
  for(let i=0;i<16;i++)for(let j=0;j<16;j++)for(let k=0;k<16;k++){
    const symbols=[i,j,k];
    const p=masses[0][i]*masses[1][j]*masses[2][k];
    all+=p;
    if(row===targetRow) {
      let cavity=cfg.uniform_symbol_prior;
      symbols.forEach((x,n)=>{if(n!==targetTone)cavity*=masses[n][x];});
      without+=cavity;
      bits.forEach(b=>{if(ones(symbols[targetTone],b))bitMass[b]+=p;});
    } else without+=p;
  }
  return {all,without,bitMass};
}
function prepare(masked) {
  const variance=varianceGrid(masked);
  const masses=cfg.H.map(H=>cfg.y.map((row,s)=>row.map((y,k)=>
    alphabet.map(x=>symbolMass(y,H,variance[s][k],x)))));
  const row=masses.map(grid=>grid.map((m,s)=>sumRow(m,s)));
  const local=masses.map((grid,id)=>{
    const m=grid[targetRow][targetTone],e=total(m),g1=bits.map(b=>total(m.filter((_,k)=>ones(k,b))));
    return {id,e,log_e:Math.log(e),g1,h1:g1.map(g=>g/e)};
  });
  return {variance,row,local};
}
function run(params,tau) {
  const {row,local}=params;
  const cavity=[0,0],full=[0,0],direct=bits.map(()=>0);
  let z=0;
  for(let first=0;first<2;first++)for(let last=0;last<2;last++){
    const path=[first,last],r=path[targetRow],row0=row[first][0],row1=row[last][1];
    const pathPrior=cfg.prior[first]*cfg.transition[first][last];
    cavity[r]+=pathPrior*(row0.without*row1.without)**(1/tau);
    full[r]+=pathPrior*(row0.all*row1.all)**(1/tau);
    const other=1-targetRow, otherRow=row[path[other]][other];
    bits.forEach(b=>{direct[b]+=pathPrior*row[r][targetRow].bitMass[b]*otherRow.all;});
    z+=pathPrior*row0.all*row1.all;
  }
  const c=normalize(cavity),w=normalize(full),e=local.map(b=>b.e);
  function decode(gate,extraExponent) {
    const effective=normalize(gate.map((p,r)=>p*e[r]**extraExponent));
    const p1=bits.map(b=>weighted(effective,local.map(r=>r.h1[b])));
    return {gate,effective,p1,hard:hard(p1)};
  }
  const raw=decode(w,1), normalized=decode(w,0), explicit=decode(c,1);
  const corrected=decode(w,1-1/tau);
  close(corrected.p1,explicit.p1,'full times e^(1-1/tau) vs cavity times e');
  close(w,normalize(c.map((p,r)=>p*e[r]**(1/tau))),'target row factor');
  close(raw.effective,normalize(c.map((p,r)=>p*e[r]**(1+1/tau))),'legacy exponent');
  if(tau===1) {
    close(explicit.p1,normalized.p1,'single count');
    close(explicit.p1,direct.map(x=>x/z),'full physical joint enumeration');
  }
  return {raw,normalized,explicit};
}
const frozen=prepare(false),masked=prepare(true),exp=fixture.expected;
close(frozen.variance,exp.frozen_variance,'frozen variance');
close(masked.variance,exp.masked_variance,'masked variance');
for(const [params,expected] of [[frozen,exp.local_frozen],[masked,exp.local_masked]]) {
  for(let r=0;r<expected.length;r++)for(const key of ['id','e','log_e','h1','g1'])
    close(params.local[r][key],expected[r][key],'branch '+r+' '+key);
}
// Log checks also resolve the very small joint densities of the masked model.
for(const [params,expected] of [[frozen,exp.local_frozen],[masked,exp.local_masked]])
  expected.forEach((branch,r)=>close(params.local[r].g1.map(Math.log),branch.g1.map(Math.log),'log joint bit density'));
close(exp.by_tau.map(row=>row.tau),[1,6],'frozen protocol temperatures');
close(cfg.tau,exp.by_tau.map(row=>row.tau),'input temperatures');
const cases=[];
for(const ref of exp.by_tau) {
  const {tau}=ref, a=run(frozen,tau), b=run(masked,tau);
  for(const [got,want,name] of [[a.raw,ref.frozen_full_raw,'full raw'],
      [a.normalized,ref.frozen_full_normalized,'full normalized'],
      [a.explicit,ref.frozen_cavity_raw,'cavity raw'],
      [b.explicit,ref.masked_refit_cavity_raw,'masked cavity raw']]) {
    for(const key of ['gate','effective','p1','hard']) close(got[key],want[key],name+' '+key);
  }
  close(ref.frozen_full_raw.target_exponent,1+1/tau,'raw exponent');
  close(ref.frozen_full_normalized.target_exponent,1/tau,'normalized exponent');
  close(ref.frozen_cavity_raw.target_exponent,1,'explicit exponent');
  close(ref.masked_refit_cavity_raw.target_exponent_within_new_model,1,'masked exponent');
  if(tau===1 && (a.raw.hard!=='0011'||a.normalized.hard!=='1111')) throw Error('strict reversal');
  if(tau===6 && (a.normalized.hard!=='0111'||a.explicit.hard!=='0011')) throw Error('temperature readout mismatch');
  cases.push({tau,full_raw:a.raw.hard,full_normalized:a.normalized.hard,
              cavity_raw:a.explicit.hard,masked_cavity_raw:b.explicit.hard});
}
console.log(JSON.stringify({status:'PASS',row_assignments_per_state:16**3,paths:4,
  numeric_checks:comparisons,max_absolute_error:worstError,cases},null,2));
