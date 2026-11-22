'use strict';
// Standalone paired synthetic experiment; estimator.cjs is task C's frozen prototype.
// Run with Node >=18: node paired_cfo.cjs /absolute/path/to/estimator.cjs > paired_results.json
const C = require(process.argv[2] || './estimator.cjs');
const S=14, N=128, SP=2, K=101, CP=6;
const used=Array.from({length:102},(_,i)=>13+i).filter(v=>v!==64);
const pilots=[13,21,29,37,45,53,61,69,77,85,93,101,109,114];
const slopes=[0,-.36,-.27,-.18,-.09,.09,.18,.27,.36];
const jumps=[-135,-90,-45,45,90,135,180].map(v=>v*Math.PI/180);
const prior=[.28,...Array(8).fill(.28/8),...Array(7).fill(.28/7),.16];
const alphabet=Array.from({length:16},(_,m)=>[(2*(m>>2)-3)/Math.sqrt(10),(2*(m%4)-3)/Math.sqrt(10)]);
const grid=(n,f)=>Array.from({length:S},(_,s)=>Array.from({length:n},(_,k)=>f(s,k)));
const product=([a,b],[c,d])=>[a*c-b*d,b*c+a*d];
const rotate=(z,p)=>product(z,[Math.cos(p),Math.sin(p)]);
const data=(s,k)=>s!==SP||!pilots.includes(used[k]);
const physical=(r,s)=>r<9?slopes[r]*(s-SP):r<16?(s===SP?0:jumps[r-9]):0;
const sum=a=>a.reduce((x,z)=>x+z,0);
const lse=a=>{const m=Math.max(...a);return m+Math.log(sum(a.map(x=>Math.exp(x-m))))};
const normalize=a=>{const c=lse(a);return a.map(x=>x-c)};
const assert=(ok,msg)=>{if(!ok)throw Error(msg)};
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const scenarios=[
  {name:'synthetic_static'}, {name:'slope_grid',om:.18},
  {name:'slope_offgrid_plus',om:.227},{name:'slope_offgrid_minus',om:-.313},
  {name:'time_cfo_ici',eps:.042},{name:'jump45',jump:45},{name:'jump90',jump:90}
];
function uniform(seed){
  let x=seed>>>0;
  return ()=>{x=(Math.imul(x,1664525)+1013904223)>>>0;return (x+.5)/4294967296};
}
// Unitary DFT reference for a single CFO perturbation of a synthetic grid.
function dft(row,sign){
  return Array.from({length:N},(_,k)=>{
    let re=0,im=0;
    for(let m=0;m<N;m++){const [a,b]=rotate(row[m],sign*2*Math.PI*k*m/N);re+=a;im+=b}
    return [re/Math.sqrt(N),im/Math.sqrt(N)];
  });
}
function injectCfo(y,eps){
  const halfShift=row=>row.slice(64).concat(row.slice(0,64));
  return y.map((row,s)=>{
    const time=dft(halfShift(row),1), block=time.slice(-CP).concat(time);
    const out=block.map((z,n)=>rotate(z,2*Math.PI*eps*((s-SP)*(N+CP)+n-CP-(N-1)/2)/N));
    return halfShift(dft(out.slice(CP),-1));
  });
}
function generate(seed,cond){
  const u=uniform(seed);
  const label=grid(N,()=>Math.floor(u()*16));
  const channel=[.89,.31];
  let y=grid(N,(s,k)=>{
    if(!used.includes(k))return [0,0];
    const j=pilots.indexOf(k);
    const x=s===SP&&j>=0?[j%4<2?-.7:.7,j%2?.7:-.7]:alphabet[label[s][k]];
    return rotate(product(channel,x),(cond.om||0)*(s-SP)+(s>=7?(cond.jump||0)*Math.PI/180:0));
  });
  if(cond.eps)y=injectCfo(y,cond.eps);
  const sd=Math.sqrt((channel[0]**2+channel[1]**2)*10**(-16/10)/2);
  y=y.map(row=>row.map(([a,b])=>{
    const mag=sd*Math.sqrt(-2*Math.log(u())),theta=2*Math.PI*u();
    return [a+mag*Math.cos(theta),b+mag*Math.sin(theta)];
  }));
  return {y,label};
}
function sharedGate(y,ctx){
  const {fold,h,hCA,v}=ctx;
  const score=grid(17,(s,r)=>{
    let total=0;
    for(let k=1-fold;k<K;k+=2){
      if(!data(s,k))continue;
      const hr=rotate(r===16?hCA[k]:h[k],physical(r,s)),z=y[s][used[k]],varz=v[s][k];
      const local=alphabet.map(x=>{
        const m=product(hr,x); return -((z[0]-m[0])**2+(z[1]-m[1])**2)/varz;
      });
      total+=lse(local)-Math.log(16*Math.PI*varz);
    }
    return total/6;
  });
  const trans=Array.from({length:17},(_,i)=>prior.map((_,j)=>Math.log(i===j?.8:.2/16)));
  const forwards=[normalize(score[0].map((v,r)=>Math.log(prior[r])+v))],backwards=Array(S);
  for(let s=1;s<S;s++)forwards.push(normalize(score[s].map((e,j)=>
    e+lse(trans.map((row,i)=>row[j]+forwards[s-1][i])))));
  backwards[S-1]=Array(17).fill(0);
  for(let s=S-2;s>=0;s--)backwards[s]=normalize(prior.map((_,i)=>
    lse(trans[i].map((t,j)=>t+score[s+1][j]+backwards[s+1][j]))));
  return forwards.map((a,s)=>normalize(a.map((z,r)=>z+backwards[s][r])));
}
function clock(f){
  const start=Date.now(), value=f();
  return [value,Date.now()-start];
}
function gateFit(y,ctx,fit){
  const f=C.evaluator(y,ctx), n=(S-1)*(ctx.fold===0?50:51);
  const gain=f(fit.omega)-f(fit.coarse),threshold=.5*Math.log(n);
  return {gain,threshold,useC:gain>threshold};
}
function foldArms(y,fold,perturb=false){
  const [[ctx,w],shared_ms]=clock(()=>{
    const ctx=C.inputs(y,pilots,fold,null);
    return [ctx,sharedGate(y,ctx)];
  });
  const nu={hLS:ctx.h,hCA:ctx.hCA,nv:ctx.v};
  const [d,d_ms]=clock(()=>C.classicalHead({coarse:0},false,w));
  const [[fit,c],c_ms]=clock(()=>{
    const f=C.fit(y,pilots,fold,nu);
    return [f,C.classicalHead(f,true,w)];
  });
  const [gate,g_ms]=clock(()=>gateFit(y,ctx,fit)),g=gate.useC?c:d;
  for(const a of [d,c,g])assert(a.logw===w&&a.lognv.every(v=>v===0),'changed gate or variance');
  assert(same(c.dphi.map(row=>row.slice(9)),d.dphi.map(row=>row.slice(9))),'changed jump or CA phase');
  if(perturb){
    const changed=y.map(row=>row.slice());
    for(let s=0;s<S;s++)for(let k=fold;k<K;k+=2)if(data(s,k))changed[s][used[k]]=[-9,13];
    const ctx2=C.inputs(changed,pilots,fold,null),w2=sharedGate(changed,ctx2);
    const f2=C.fit(changed,pilots,fold,{hLS:ctx2.h,hCA:ctx2.hCA,nv:ctx2.v});
    assert(same(ctx2,ctx)&&same(w,w2)&&same(f2.cells,fit.cells)&&same(gateFit(changed,ctx2,f2),gate),
      'target-fold leakage');
  }
  const [D,demD]=clock(()=>C.demap(y,ctx,d));
  const [Cout,demC]=clock(()=>C.demap(y,ctx,c));
  const [G,demG]=clock(()=>C.demap(y,ctx,g));
  return {D,C:Cout,G,omega:fit.omega,coarse:fit.coarse,gate,n_evaluations:fit.evaluations,
    timing:{shared_ms,d_head_ms:d_ms,c_head_ms:c_ms,g_gate_ms:g_ms,d_demap_ms:demD,
      c_demap_ms:demC,g_demap_ms:demG,D_ms:shared_ms+d_ms+demD,C_ms:shared_ms+c_ms+demC,
      G_ms:shared_ms+c_ms+d_ms+g_ms+demG}};
}
const methodNames=['D','C','G'],pairNames=['DC','CG','DG'];
function blank(){
  return {n:0,err:{D:0,C:0,G:0},nll_sum:{D:0,C:0,G:0},
    pairs:Object.fromEntries(pairNames.map(p=>[p,{new_errors:0,rescues:0}]))};
}
function scoreBit(m,truth,llrs){
  m.n++;
  const bad={};
  for(const n of methodNames){
    const ll=llrs[n],signed=truth?ll:-ll;
    assert(Number.isFinite(ll),'nonfinite LLR');
    bad[n]=Number((ll>0)!==Boolean(truth));
    m.err[n]+=bad[n];
    m.nll_sum[n]+=Math.max(-signed,0)+Math.log1p(Math.exp(-Math.abs(signed)));
  }
  for(const [p,v] of Object.entries(m.pairs)){
    const a=bad[p[0]],b=bad[p[1]]; v.new_errors+=Number(!a&&b);v.rescues+=Number(a&&!b);
  }
}
function scoreFrame(frame,folds){
  const whole=blank(),pre=blank(),post=blank(),seen=new Set();
  for(const f of folds)for(let j=0;j<f.D.length;j++){
    const rd=f.D[j],key=rd.s*128+rd.k;
    assert(!seen.has(key),'duplicate target');seen.add(key);
    for(const n of methodNames)assert(f[n][j].s===rd.s&&f[n][j].k===rd.k,'target coordinates differ');
    for(let b=0;b<4;b++){
      const truth=(frame.label[rd.s][rd.k]>>(3-b))&1;
      const ll=Object.fromEntries(methodNames.map(n=>[n,f[n][j].llr[b]]));
      scoreBit(whole,truth,ll);scoreBit(rd.s<7?pre:post,truth,ll);
    }
  }
  assert(seen.size===1400&&whole.n===5600&&pre.n+post.n===whole.n,'coverage');
  return {whole,pre,post};
}
function runFrame(seed,scen,check=false){
  const frame=generate(seed,scen),folds=[0,1].map(f=>foldArms(frame.y,f,check));
  const timing=Object.fromEntries(Object.keys(folds[0].timing).map(k=>[k,sum(folds.map(f=>f.timing[k]))]));
  return {seed,condition:scen.name,metrics:scoreFrame(frame,folds),timing,
    folds:folds.map(({omega,coarse,gate,n_evaluations})=>({omega,coarse,gate,n_evaluations}))};
}
function aggregate(rows,part='whole'){
  const m=blank();
  for(const row of rows){
    const z=row.metrics[part];m.n+=z.n;
    for(const n of methodNames){m.err[n]+=z.err[n];m.nll_sum[n]+=z.nll_sum[n]}
    for(const p of pairNames)for(const k of ['new_errors','rescues'])m.pairs[p][k]+=z.pairs[p][k];
  }
  for(const [p,v] of Object.entries(m.pairs)){
    v.flips=v.new_errors+v.rescues;v.net_errors=v.new_errors-v.rescues;
    assert(v.net_errors===m.err[p[1]]-m.err[p[0]],'paired counts inconsistent');
    v.worse_frames=rows.filter(r=>r.metrics[part].err[p[1]]>r.metrics[part].err[p[0]]).length;
    v.better_frames=rows.filter(r=>r.metrics[part].err[p[1]]<r.metrics[part].err[p[0]]).length;
  }
  m.ber=Object.fromEntries(methodNames.map(n=>[n,m.err[n]/m.n]));
  m.nll=Object.fromEntries(methodNames.map(n=>[n,m.nll_sum[n]/m.n]));
  return m;
}
const choose=z=>z.err.D<=z.err.C?'D':'C';
function selectDev(dev){
  return {global:choose(aggregate(dev)),
    stratum:Object.fromEntries(scenarios.map(sc=>[sc.name,choose(aggregate(dev.filter(z=>z.condition===sc.name)))]))};
}
function pooledTiming(rows){
  const out={};for(const k of Object.keys(rows[0].timing)){
    const sorted=rows.map(z=>z.timing[k]).sort((a,b)=>a-b),n=sorted.length;
    out[k]={median:(sorted[Math.floor((n-1)/2)]+sorted[Math.floor(n/2)])/2,p95:sorted[Math.ceil(.95*n)-1]};
  }
  return out;
}
function run(){
  runFrame(55000,scenarios[0],true); // Includes held-out-fold mutation check. Excluded from timing/results.
  const dev=[54001,54002].flatMap(seed=>scenarios.map(sc=>runFrame(seed,sc)));
  const fixed=selectDev(dev);  // Frozen before generating any held-out frame.
  const test=[53001,53002].flatMap(seed=>scenarios.map(sc=>runFrame(seed,sc)));
  const layers=scenarios.map(sc=>{
    const rows=test.filter(r=>r.condition===sc.name),z=aggregate(rows),sel=fixed.stratum[sc.name];
    return {name:sc.name,dev_errors:aggregate(dev.filter(r=>r.condition===sc.name)).err,
      layer_fixed:sel,global_fixed:fixed.global,gate_active_folds:sum(rows.flatMap(r=>r.folds.map(f=>+f.gate.useC))),
      ...z,layer_fixed_to_gate:z.pairs[sel+'G'],global_fixed_to_gate:z.pairs[fixed.global+'G'],
      ...(sc.jump?{before_jump:aggregate(rows,'pre'),after_jump:aggregate(rows,'post')}:{}),
      timing_ms:pooledTiming(rows)};
  });
  return {format:1,dev_seeds:[54001,54002],test_seeds:[53001,53002],fixed,
    global_dev_errors:aggregate(dev).err,dev_frames:dev,test_frames:test,layers,all_test_timing_ms:pooledTiming(test),
    note:'Each condition reuses the same two test labels and noise realizations. Timing uses Date.now(), ms, serial V8.'};
}
module.exports={run,runFrame,aggregate,selectDev,generate,foldArms,scenarios};
if(require.main===module)console.log(JSON.stringify(run(),null,2));
