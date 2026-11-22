'use strict';
// Independent oracle. Uses probability arithmetic, state-path enumeration and
// target-symbol sums. The Python receiver uses log-domain forward/backward messages.
function verify(fixture, python, A, ab) {
  const result = {discrete:{max_abs:{},count:0,candidate_bits:0,assignments:0},
                  qam:{max_abs:{},candidate_bits:0,bit_decisions:0}};
  function close(group,key,a,b,tol=2e-12){
    const d=Math.abs(a-b);
    if(!Number.isFinite(d)||d>tol)throw Error(key+': '+a+' vs '+b);
    group.max_abs[key]=Math.max(group.max_abs[key]||0,d);
  }
  const sum=a=>a.reduce((p,q)=>p+q,0), norm=a=>a.map(x=>x/sum(a));
  const fraction=s=>{const [n,d]=s.split('/').map(Number);return n/d;};
  const hard=p=>Number(p>.5);
  const allcases=A.otherReferences.concat([{y:A.reference.y,p1:A.reference.bits.map(b=>b.p1),map:A.reference.bits.map(b=>b.map)}]);
  if(python.discrete.length!==allcases.length)throw Error('case count');
  for(let f=0;f<allcases.length;f++){
    const {y,p1,map}=allcases[f],N=y.length,S=N/2, d=result.discrete;
    const cnt=Array.from({length:N},()=>[[0n,0n,0n,0n],[0n,0n,0n,0n]]);
    let total=0n;
    const ones=[[1,2,6,9],[8,5,2,1]], transition=[[9,1],[3,7]];
    for(let path=0;path<(1<<S);path++){
      const r=Array.from({length:S},(_,s)=>(path>>s)&1);
      let prior=BigInt(r[0]===0?6:4);
      for(let s=1;s<S;s++)prior*=BigInt(transition[r[s-1]][r[s]]);
      for(let word=0;word<4**N;word++){
        let w=prior;
        for(let n=0;n<N;n++){
          const x=(word>>(2*n))&3, z=ones[r[n>>1]][x];w*=BigInt(y[n]?z:10-z);
        }
        total+=w;d.assignments++;
        for(let n=0;n<N;n++)cnt[n][r[n>>1]][(word>>(2*n))&3]+=w;
      }
    }
    for(let n=0;n<N;n++){
      const local=ones.map(row=>row.map(v=>y[n]?v:10-v));
      const ell=local.map(g=>sum(g)/40);
      const full=cnt[n].map(row=>Number(row.reduce((a,v)=>a+v,0n))/Number(total));
      const cavity=norm(full.map((v,r)=>v/ell[r]));
      const eff0=norm(full.map((v,r)=>v*ell[r]));
      const [orig,corr]=python.discrete[f][n];
      for(let b=0;b<2;b++){
        const branch=cnt[n].map(row=>Number(row.reduce((a,q,x)=>a+(((x>>(1-b))&1)?q:0n),0n))/Number(row.reduce((a,v)=>a+v,0n)));
        const truth=fraction(p1[2*n+b]), repeated=sum(branch.map((v,r)=>v*eff0[r]));
        close(d,'A_published_fraction',truth,sum(branch.map((v,r)=>v*full[r])));
        close(d,'corrected_p1',corr.p1[b],truth);close(d,'original_p1',orig.p1[b],repeated);
        if(corr.hard[b]!==map[2*n+b]||orig.hard[b]!==hard(repeated))throw Error('A hard');
        for(let r=0;r<2;r++){
          close(d,'full',orig.weight[r],full[r]);close(d,'cavity',corr.weight[r],cavity[r]);
          close(d,'log_marginal',orig.ell[r],Math.log(ell[r]));
          close(d,'branch_p1',corr.branch_bit[r][b],branch[r]);d.candidate_bits++;
        }
        d.count++;
      }
    }
  }

  const qam=Array.from({length:16},(_,i)=>[(2*(i>>2)-3)/Math.sqrt(10),(2*(i%4)-3)/Math.sqrt(10)]);
  const frames=fixture.frames.concat([fixture.unlabelled]);
  if(frames.length!==python.qam.length)throw Error('QAM frames');
  for(let j=0;j<frames.length;j++){
    const f=frames[j],out=python.qam[j];
    const [S,K]=[f.y.length,f.y[0].length];
    if(out.id!==f.id||out.stratum!==f.stratum)throw Error('frame identity');
    for(const entry of out.entries){
      const [s0,k0]=entry.target,t=entry.tau;
      for(const mode of ['original','explicit','rebuild']){
        // Gather actual footprint multiplicities by enumerating offsets. Exclusion
        // drops both the energy and its duplicated edge counts, never zero-fills.
        const nv=f.y.map((row,s)=>row.map((_,k)=>{
          let a=[];
          for(let u=s-1;u<=s+1;u++)for(let v=k-1;v<=k+1;v++){
            const p=Math.max(0,Math.min(S-1,u)),q=Math.max(0,Math.min(K-1,v));
            if(mode==='rebuild'&&p===s0&&q===k0)continue;
            const z=f.y[p][q];a.push(z[0]**2+z[1]**2);
          }
          const e=sum(a)/a.length;return e/1.12>1.75?Math.min(20,Math.max(.12,e-1)):.12;
        }));
        const tab=f.y.map((row,s)=>row.map(([a,b],k)=>f.h.map(([u,v])=>qam.map(([x,y])=>{
          const d=(a-u*x+v*y)**2+(b-u*y-v*x)**2;
          return Math.exp(-d/nv[s][k])/(Math.PI*nv[s][k]*16);
        }))));
        const joint=tab[s0][k0], m=joint.map(sum);
        // Enumerate all four paths. Collapse nuisance symbols at each non-target
        // site by direct probability summation. At tau=1 this is exact marginalization.
        // tau=6 explicitly scores product(marginals)^(1/tau), not averaged powered symbols.
        const gate=[0,0],symbols=Array(16).fill(0);
        for(let i=0;i<2;i++)for(let u=0;u<2;u++){
          const path=[i,u],r=path[s0];let w=f.prior[i]*f.transition[i][u];
          for(let s=0;s<S;s++)for(let k=0;k<K;k++){
            if(mode!=='original'&&s===s0&&k===k0)continue;
            w*=sum(tab[s][k][path[s]])**(1/t);
          }
          gate[r]+=w;
          for(let x=0;x<16;x++)symbols[x]+=w*joint[r][x];
        }
        const scaled=norm(symbols), pred=entry.mode[mode],d=result.qam;
        const state=norm(gate),effective=norm(gate.map((w,r)=>w*m[r]));
        for(let s=0;s<S;s++)for(let k=0;k<K;k++)close(d,'variance',pred.variance[s][k],nv[s][k]);
        for(let b=0;b<4;b++){
          const ids=qam.map((_,i)=>i).filter(i=>(i>>(3-b))&1);
          const p=sum(ids.map(i=>scaled[i]));
          close(d,'p1',pred.p1[b],p);
          if(pred.hard[b]!==hard(p))throw Error('QAM decision');
          for(let r=0;r<2;r++){
            close(d,'weight',pred.weight[r],state[r]);close(d,'effective',pred.effective[r],effective[r]);
            close(d,'ell',pred.ell[r],Math.log(m[r]),1e-10);
            close(d,'branch_p1',pred.branch_bit[r][b],sum(ids.map(i=>joint[r][i]))/m[r]);
            d.candidate_bits++;
          }
          d.bit_decisions++;
        }
        if(t===1&&mode==='explicit'){
          const orig=entry.mode.original;
          for(let b=0;b<4;b++)close(d,'tau1_full_conditional_vs_cavity',pred.p1[b],
            sum(orig.weight.map((w,r)=>w*orig.branch_bit[r][b])));
        }
      }
    }
  }
  // The frozen A/B handoff includes the independent A derivation and values
  // rerun by B. Compare the exact reduced IQ, H, transition and bit order first.
  const single=fixture.unlabelled,cfg=ab.input;
  for(const [x,y] of [[single.y,cfg.y],[single.h,cfg.H],[single.prior,cfg.prior],[single.transition,cfg.transition]])
    if(JSON.stringify(x)!==JSON.stringify(y))throw Error('different A/B inputs');
  result.shared_fixture={max_abs:{},checks:0};const g=result.shared_fixture;
  function same(a,b,label){
    if(Array.isArray(b)){if(!Array.isArray(a)||a.length!==b.length)throw Error(label);b.forEach((v,i)=>same(a[i],v,label));}
    else{close(g,label,a,b,2e-12);g.checks++;}
  }
  for(const row of ab.expected.by_tau){
    const sample=python.qam[python.qam.length-1].entries.find(z=>z.tau===row.tau);
    if(JSON.stringify(sample.target)!==JSON.stringify(cfg.target))throw Error('different target');
    const modes=sample.mode;
    for(const [src,dst] of [['original','frozen_full_raw'],['explicit','frozen_cavity_raw'],['rebuild','masked_refit_cavity_raw']]){
      const rec=modes[src],f=row[dst],loc=src==='rebuild'?ab.expected.local_masked:ab.expected.local_frozen;
      same(rec.variance,src==='rebuild'?ab.expected.masked_variance:ab.expected.frozen_variance,'variance');
      same(rec.weight,f.gate,'gate');same(rec.effective,f.effective,'effective');same(rec.p1,f.p1,'p1');
      same(rec.ell,loc.map(x=>x.log_e),'ell');same(rec.branch_bit,loc.map(x=>x.h1),'branch_bit');
      if(rec.hard.join('')!==f.hard)throw Error('hand-off MAP mismatch');
    }
    const o=modes.original;
    const fullNormalized=[0,1,2,3].map(b=>sum(o.weight.map((p,r)=>p*o.branch_bit[r][b])));
    same(fullNormalized,row.frozen_full_normalized.p1,'full_normalized_p1');
    if(fullNormalized.map(hard).join('')!==row.frozen_full_normalized.hard)throw Error('normalized MAP');
  }
  return result;
}

if(typeof module!=='undefined'){
  module.exports={verify};
  if(require.main===module){
    const fs=require('node:fs');
    const [fixture,raw,aReference,ab]=process.argv.slice(2).map(f=>{
      let bytes=fs.readFileSync(f);
      if(f.endsWith('.gz.b64'))bytes=Buffer.from(bytes.toString('ascii'),'base64');
      if(f.endsWith('.gz')||f.endsWith('.gz.b64'))bytes=require('node:zlib').gunzipSync(bytes);
      return JSON.parse(bytes.toString('utf8'));
    });
    console.log(JSON.stringify(verify(fixture,raw,aReference,ab),null,2));
  }
}
