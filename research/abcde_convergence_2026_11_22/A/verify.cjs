// Reproduce with node verify.cjs (Node >= 14). No packages, network or randomness.
"use strict";
const same = (ok, why) => { if (!ok) throw Error(why); };
const sum = xs => xs.reduce((a,b) => a+b, 0n);
const prod = xs => xs.reduce((a,b) => a*b, 1n);
function fraction(n,d) {
  same(d > 0n, "zero/negative denominator");
  let a = n < 0n ? -n : n, b = d;
  while (b) [a,b] = [b,a%b];
  return String(n/a) + "/" + String(d/a);
}
function* words(length, base) {
  if (length === 0) { yield []; return; }
  for (const prefix of words(length-1,base))
    for(let k=0;k<base;++k) yield [...prefix,k];
}
// The two states are enumerated in order 0,1. Symbols 00,01,10,11 are uniform.
// Numerators: initial /10, transition /10, binary emission /10; all exact.
const initial=[6n,4n], transition=[[9n,1n],[3n,7n]];
const ones=[[1n,2n,6n,9n],[8n,5n,2n,1n]];
function emit(r,x,y) { return y===1 ? ones[r][x] : 10n-ones[r][x]; }
function oracle(Y) {
  const T=Y.length/2, onesMass=Array(2*Y.length).fill(0n);
  let total=0n, assignments=0;
  for(const path of words(T,2)) {
    let pathMass=initial[path[0]];
    for(let t=1;t<T;t++) pathMass*=transition[path[t-1]][path[t]];
    for(const X of words(Y.length,4)) {
      let joint=pathMass;
      for(let i=0;i<Y.length;i++) joint*=emit(path[i>>1],X[i],Y[i]);
      total+=joint; assignments++;
      for(let i=0;i<X.length;i++) {
        if(X[i]&2) onesMass[2*i]+=joint;
        if(X[i]&1) onesMass[2*i+1]+=joint;
      }
    }
  }
  return {total,onesMass,assignments};
}
// An independent calculation marginalizes symbols first, then sends two HMM messages.
// The oracle above never calls these functions and visits each full assignment.
function messages(Y) {
  const T=Y.length/2, states=[0,1], symbols=[0,1,2,3];
  const local=Y.map(y=>states.map(r=>symbols.map(x=>emit(r,x,y))));
  const site=local.map(rows=>rows.map(sum));
  const row=Array.from({length:T},(_,t)=>states.map(r=>site[2*t][r]*site[2*t+1][r]));
  const left=[initial], right=Array(T);
  for(let t=1;t<T;t++) left[t]=states.map(r=>
    sum(states.map(s=>left[t-1][s]*row[t-1][s]*transition[s][r])));
  right[T-1]=[1n,1n];
  for(let t=T-2;t>=0;t--) right[t]=states.map(r=>
    sum(states.map(s=>transition[r][s]*row[t+1][s]*right[t+1][s])));
  return {local,site,left,right};
}
function check(Y) {
  const ref=oracle(Y), {local,site,left,right}=messages(Y), bits=[];
  for(let i=0;i<Y.length;i++) {
    const t=i>>1, states=[0,1], cavity=states.map(r=>left[t][r]*right[t][r]*site[i^1][r]);
    const e=site[i], full=states.map(r=>cavity[r]*e[r]);
    for(const [bit,mask] of [[0,2],[1,1]]) {
      const g=local[i].map(scores=>sum(scores.filter((_,x)=>(x&mask)!==0)));
      const cavityP=fraction(sum(states.map(r=>cavity[r]*g[r])),sum(full));
      // Form the mixture of normalized branch probabilities with a common denominator.
      const common=prod(e);
      const fullP=fraction(sum(states.map(r=>full[r]*g[r]*(common/e[r]))),sum(full)*common);
      const expected=fraction(ref.onesMass[2*i+bit],ref.total);
      const wrong=fraction(sum(states.map(r=>full[r]*g[r])),sum(states.map(r=>full[r]*e[r])));
      same(cavityP===expected && fullP===expected,"oracle mismatch "+[Y,i,bit]);
      same(sum(full)===ref.total,"absolute joint evidence mismatch");
      const d1=sum(states.map(r=>cavity[r]*(2n*g[r]-e[r])));
      const d2=sum(states.map(r=>full[r]*(2n*g[r]-e[r])));
      const [qn,qd]=wrong.split("/").map(BigInt);
      const exactMargin=2n*ref.onesMass[2*i+bit]-ref.total;
      same((d1*d2<0n)===(exactMargin*(2n*qn-qd)<0n),"strict reversal equivalence");
      bits.push({row:t+1,site:i%2+1,bit:bit===0?"MSB":"LSB",p1:expected,
                 map:2n*ref.onesMass[2*i+bit]>ref.total?1:0,duplicate_p1:wrong,
                 branches:states.map(r=>({state:r,cavity_weight:fraction(cavity[r],sum(cavity)),
                    full_weight:fraction(full[r],sum(full)),target_evidence:fraction(e[r],40n),
                    target_joint_bit1:fraction(g[r],40n),branch_p1:fraction(g[r],e[r])}))});
    }
  }
  return {y:Y,assignments:ref.assignments,bits};
}
// Hand-designed joint laws: event y=1 has state evidence (1/5,4/5);
// conditional on (r,y=1), bit-one probability is h[r]/10.
// Complete y=0 with any bit distribution (e.g. fair) to get a joint law.
function examples() {
  const examples=[];
  for(const [name,h,p,q] of [
    ["no_flip",[9n,6n],"3/4","33/50"],["flip",[9n,2n],"11/20","17/50"],
    ["true_tie",[8n,2n],"1/2","8/25"],["duplicate_tie",[9n,4n],"13/20","1/2"]]) {
    const a=[4n,1n], e=[1n,4n], w=a.map((v,r)=>v*e[r]);
    const w2=w.map((v,r)=>v*e[r]);
    const posterior=weights=>fraction(sum(weights.map((v,r)=>v*h[r])),10n*sum(weights));
    same(posterior(w)===p && posterior(w2)===q,name);
    examples.push({name,p1:p,duplicate_p1:q});
  }
  return examples;
}
function edgeChecks() {
  const e=[0n,4n], a=[2n,1n], joint1=[0n,3n];
  const support=e.map((v,r)=>r).filter(r=>a[r]*e[r]>0n);
  same(support.length===1 && fraction(joint1[support[0]],e[support[0]])==="3/4","zero branch");
  let rejected=false; try {fraction(0n,0n);} catch {rejected=true;}
  same(rejected,"zero total evidence must be rejected");
  // Finite temperature: first marginalize symbols, then square, versus square emissions first.
  const k=[1n,3n]; // actual emission masses 1/4, 3/4 with a fair symbol prior
  same(fraction(sum(k)**2n,64n)==="1/4" && fraction(sum(k.map(v=>v*v)),32n)==="5/16","temperature");
  // Observable-dependent Gaussian scale: mu=1, variance 1 on y<0 and 4 on y>=0.
  // Numerically integrate the proposed density without calling a Gaussian CDF.
  const normal=(y,mu,v)=>Math.exp(-((y-mu)**2)/(2*v))/Math.sqrt(2*Math.PI*v);
  let integral=0; const step=0.002;
  for(let j=0;j<20000;j++) {const y=-20+(j+0.5)*step; integral+=step*normal(y,1,y<0?1:4);}
  same(Math.abs(integral-0.850117715)<1e-6,"observation-dependent variance");
  // Remaining observation y2=0, state means 0 and 1, prior fair.
  // v=1+mean(y^2); target y1=2. Return odds for state 0 over state 1.
  const deleted=Math.exp(1/(2*3)), masked=Math.exp(1/(2*1));
  same(deleted<masked,"deletion vs refit");
  return {zero_branch_p1:"3/4",zero_total_rejected:rejected,temperature_after_sum:"1/4",
          temperature_before_sum:"5/16",gaussian_mass_midpoint:integral,
          odds_delete_frozen:deleted,odds_refit_masked:masked};
}
const otherReferences=[];
let observations=0, bitChecks=0, assignments=0;
for(const n of [2,4]) for(const y of words(n,2)) {
  const a=check(y); otherReferences.push({y,p1:a.bits.map(b=>b.p1),map:a.bits.map(b=>b.map)}); observations++; bitChecks+=a.bits.length; assignments+=a.assignments;
}
const selected=check([1,0,0,1,1,1]);
const expectedBits=[
  "48108429/80872582","42371561/80872582","98235607/242617746","114862003/242617746",
  "90051727/242617746","111897763/242617746","51770769/80872582","43710481/80872582",
  "54310413/80872582","44638953/80872582","54310413/80872582","44638953/80872582"];
same(selected.bits.every((b,i)=>b.p1===expectedBits[i]),"three-row frozen references");
observations++; bitChecks+=selected.bits.length; assignments+=selected.assignments;
const output={commit:"f566f9b8b0ddfb468a4cb3fd64588bfa46c87212",observations,assignments,bitChecks,
   examples:examples(),edge:edgeChecks(),reference:selected,otherReferences};
console.log(JSON.stringify(output,null,2));
