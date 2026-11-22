/* Node 18+; run: node mini.js. Indices and bit positions start at zero.
   Exact tiny path enumeration. This is a reduced model, not the 17-state receiver. */
'use strict';
const sum=a=>a.reduce((x,y)=>x+y,0);
const lse=a=>{const m=Math.max(...a);return m+Math.log(sum(a.map(x=>Math.exp(x-m))))};
const soft=a=>{const m=Math.max(...a),q=a.map(x=>Math.exp(x-m)),z=sum(q);return q.map(v=>v/z)};
const avg=a=>sum(a)/a.length;
const clip=(x,n)=>Math.max(0,Math.min(n-1,x));
const abs2=([a,b])=>a*a+b*b;
const eq=(a,b,t=1e-12)=>a.every((v,i)=>Math.abs(v-b[i])<t);
const pam=[-3,-1,1,3].map(v=>v/Math.sqrt(10));
const qam=pam.flatMap(i=>pam.map(q=>[i,q]));
const qpsk=[[1,1],[1,-1],[-1,1],[-1,-1]].map(p=>p.map(v=>v*.707));
function local(y,[a,b],v,alphabet,dc=[0,0]){
  const joint=alphabet.map(([i,q])=>-abs2([y[0]-dc[0]-a*i+b*q,y[1]-dc[1]-b*i-a*q])/v-Math.log(Math.PI*v*alphabet.length));
  const marginal=lse(joint), cond=soft(joint), n=Math.log2(alphabet.length);
  const bit=Array.from({length:n},(_,j)=>sum(cond.filter((_,k)=>(k>>(n-1-j))&1)));
  return {joint,marginal,bit};
}
function combine(c,base,raw){
  const effective=raw?soft(base.map((v,r)=>Math.log(v)+c[r].marginal)):base;
  const bit=c[0].bit.map((_,b)=>Math.min(1,Math.max(0,sum(c.map((e,r)=>e.bit[b]*effective[r])))));
  return {candidate:c.map((v,r)=>({id:r,local_log_marginal:v.marginal,weight:base[r],effective:effective[r],branch_bit:v.bit})),bit,hard:bit.map(x=>Number(x>0.5))};
}
// 2 rows x 3 carriers. On the small grid, the pilot-derived floor and signal power are known.
function ofdm(y,tau=1){
  const s0=0,k0=1, h=[[1,0],[.59,.53]], n0=.12;
  const rows=y.map((row,s)=>row.map((_,k)=>[s,k]));
  const output={target:[s0,k0],tau};
  for (const mode of ['original','explicit','rebuild']){
    const retain=(s,k)=>!(s===s0&&k===k0&&mode==='rebuild');
    function variance(s,k){
      const energies=[];
      for(let ds=-1;ds<=1;ds++)for(let dk=-1;dk<=1;dk++){
        const a=clip(s+ds,2), b=clip(k+dk,3);
        if(retain(a,b)) energies.push(abs2(y[a][b]));
      }
      const energy=avg(energies);return energy/(1+n0)>1.75?Math.min(20,Math.max(n0,energy-1)):n0;
    }
    const nv=rows.map(row=>row.map(([s,k])=>variance(s,k)));
    const tab=h.map(hr=>rows.map(row=>row.map(([s,k])=>local(y[s][k],hr,nv[s][k],qam))));
    const score=tab.map(t=>t.map((row,s)=>sum(row.map((z,k)=>
      (s===s0&&k===k0&&mode!=='original')?0:z.marginal))));
    // Enumerate all four candidate paths; keep initial state (the target row).
    const logw=h.map((_,r)=>lse(h.map((_,u)=>Math.log([.54,.46][r])+Math.log(u===r?.82:.18)+(score[r][0]+score[u][1])/tau)));
    const feat=rows.map(row=>avg(row.filter(([s,k])=>retain(s,k)).map(([s,k])=>abs2(y[s][k]))));
    const c=tab.map(t=>t[s0][k0]);
    output[mode]={nv,row_energy_feature:feat,score,...combine(c,soft(logw),true)};
    if(mode==='original')output.normalized_branch=combine(c,soft(logw),false);
  }
  if(tau===1&&!eq(output.explicit.bit,output.normalized_branch.bit))throw Error('cavity identity');
  return output;
}
function pluto(){
  const y=[[.9,-.4],[.3,.6]], hs=[[.8,.15],[-.2,.65]], vs=[.23,.37], prior=[.61,.39];
  const c=hs.map((h,r)=>y.map(z=>local(z,h,vs[r],qpsk,[.04,-.02])));
  const w=soft(c.map((a,r)=>Math.log(prior[r])+sum(a.map(x=>x.marginal))));
  const cavity=soft(c.map((a,r)=>Math.log(prior[r])+a[1].marginal));
  const target=c.map(a=>a[0]), b=combine(target,w,false);
  const direct=soft(qpsk.map((_,i)=>lse(hs.flatMap((_,r)=>qpsk.map((_,j)=>Math.log(prior[r])+c[r][0].joint[i]+c[r][1].joint[j])))));
  const directBits=[direct[2]+direct[3],direct[1]+direct[3]];
  if(!eq(directBits,b.bit)||!eq(b.bit,combine(target,cavity,true).bit))throw Error('PLUTO joint enumeration');
  return {target_payload_index:0,frame_index:26,B3a:combine(target,prior,true),B3b:b,cavity,directBits};
}
const y=[[[.35,.25],[-.88,2.63],[.5,-.3]],[[-.2,.8],[.95,.33],[-.48,-.58]]];
const o=ofdm(y),changed=JSON.parse(JSON.stringify(y));changed[0][1]=[-1.5,.1];
const p=ofdm(changed);
for(const key of ['nv','row_energy_feature','score'])if(JSON.stringify(o.rebuild[key])!==JSON.stringify(p.rebuild[key]))throw Error('target enters rebuild: '+key);
if(eq(o.original.bit,o.explicit.bit)||eq(o.explicit.bit,o.rebuild.bit))throw Error('probe lacks separation');
console.log(JSON.stringify({OFDM:o,OFDM_tau2:ofdm(y,2),PLUTO:pluto(),target_perturbation_rebuild_invariance:true},null,2));
