'use strict';
// Four deterministic engineering fixtures, no repository data or label input to fitting.
const C = require('./estimator.cjs');
const {S,N,CP,SP,K,R,USED,P,X,data,physical} = C.layout;
const {lse,mul,rot} = C.math;
const LL_TEMP=6, P_STAY=.8; // pinned Dev calibration, used by repository Group C
const V = .03; // predeclared shared receiver variance for ALL cells, incl. no added noise
const eps = .042, omega = 2*Math.PI*eps*(N+CP)/N;
const grid = (m,f) => Array.from({length:S},(_,s)=>Array.from({length:m},(_,k)=>f(s,k)));
const prior = [.28,...Array(8).fill(.28/8),...Array(7).fill(.28/7),.16];
const normalize = v => {const z=lse(v);return v.map(x=>x-z)};
function makeContext(y,fold){
  // Pilot-only LS. Historical path duplicates LS in this synthetic fixture only.
  const {h,hCA} = C.inputs(y,P,fold,null);
  const nv=grid(K,()=>V);
  const emit=grid(R,()=>0);
  for(let s=0;s<S;s++)for(let k=0;k<K;k++){
    if(k%2===fold || !data(s,k))continue;
    const z=y[s][USED[k]];
    for(let r=0;r<R;r++){
      const a=rot(r===16?hCA[k]:h[k],physical(s,r));
      emit[s][r]+=lse(X.map(x=>{
        const p=mul(a,x);return -((p[0]-z[0])**2+(p[1]-z[1])**2)/V;
      }))/LL_TEMP; // common log(16*pi*V) terms cancel.
    }
  }
  const logT = (i,j) => Math.log(i===j?P_STAY:(1-P_STAY)/(R-1));
  const fw=[],bw=Array(S);
  fw[0]=normalize(prior.map((p,r)=>Math.log(p)+emit[0][r]));
  for(let s=1;s<S;s++)fw[s]=normalize(prior.map((_,j)=>emit[s][j]+lse(fw[s-1].map((v,i)=>v+logT(i,j)))));
  bw[S-1]=Array(R).fill(0);
  for(let s=S-2;s>=0;s--)bw[s]=normalize(prior.map((_,i)=>lse(bw[s+1].map((v,j)=>v+emit[s+1][j]+logT(i,j)))));
  const logw=fw.map((v,s)=>normalize(v.map((x,r)=>x+bw[s][r])));
  return {hLS:h,hCA,nv,logw};
}
function runFold(y,fold,ctx,method){
  const packet=C.headPacket({y,visiblePilots:P,fold,nuisance:ctx,method,logw:ctx.logw});
  const shared={h:ctx.hLS,hCA:ctx.hCA,v:ctx.nv,fold};
  return {packet, target:C.demap(y,shared,packet)};
}
function examples(){
  const raw=C.fixture(0), ideal=C.fixture(omega).y;
  const ici=C.addCfo(raw.y,eps,false), noisy=C.addCfo(raw.y,eps,true);
  // Subtract identical deterministic noiseless signal to reuse exactly the same noise draw.
  const idealNoisy=grid(N,(s,k)=>ideal[s][k].map((x,i)=>x+(noisy[s][k][i]-ici[s][k][i])));
  return {truth:raw.truth, signals:{cpe:ideal,cpe_awgn:idealNoisy,td_cfo:ici,td_cfo_awgn:noisy}};
}
function assert(ok,msg){if(!ok)throw new Error(msg)}
function ablate(){
  const {truth,signals}=examples(), results={};
  for(const [scene,y] of Object.entries(signals)){
    const er={discrete:0,continuous:0}, estimates=[];
    const visited=new Set();
    for(let f=0;f<2;f++){
      const ctx=makeContext(y,f);
      const d=runFold(y,f,ctx,'discrete'),c=runFold(y,f,ctx,'continuous');
      assert(JSON.stringify(d.packet.logw)===JSON.stringify(c.packet.logw),'weights differ');
      assert([...d.packet.lognv,...c.packet.lognv].every(v=>v===0),'variance differs');
      for(let i=0;i<d.target.length;i++){
        const a=d.target[i],b=c.target[i],id=a.s*N+a.k;
        assert(!visited.has(id)&&a.s===b.s&&a.k===b.k,'duplicate/mismatched coordinate');
        visited.add(id);
        for(const [name,re] of [['discrete',a],['continuous',b]])for(let bit=0;bit<4;bit++){
          assert(Number.isFinite(re.llr[bit]),'nonfinite LLR');
          const label=(truth[a.s][a.k]>>(3-bit))&1;
          er[name]+=Number((re.llr[bit]>0)!==Boolean(label));
        }
      }
      estimates.push(c.packet.omega);
      if(scene==='td_cfo_awgn'){
        const mutated=y.map(row=>row.slice());
        for(let s=0;s<S;s++)for(let k=f;k<K;k+=2)if(data(s,k))mutated[s][USED[k]]=[11,-23];
        const other=makeContext(mutated,f);
        assert(JSON.stringify(other)===JSON.stringify(ctx),'target influenced nuisance/HMM');
        for(const [method,old] of [['discrete',d],['continuous',c]]){
          const updated=runFold(mutated,f,other,method);
          assert(JSON.stringify(updated.packet)===JSON.stringify(old.packet),'target influenced head');
          assert(updated.target.some((re,i)=>Math.abs(re.llr[0]-old.target[i].llr[0])>1e-6),'demapper ignored target');
        }
      }
    }
    assert(visited.size===1400,'incomplete coverage');
    results[scene]={errors:er,bits:4*visited.size,omega_hat:estimates};
  }
  return {synthetic_only:true,eps,omega,variance:V,ll_temp:LL_TEMP,p_stay:P_STAY,matched_17way:results,
    target_intervention:'all target REs replaced, both folds, both heads, full context recomputed'};
}
function nativeFixture(){
  const y=examples().signals.td_cfo_awgn,contexts=[], expected={discrete:[],continuous:[]};
  for(let fold=0;fold<2;fold++){
    const ctx=makeContext(y,fold);contexts.push(ctx);
    for(const method of Object.keys(expected))expected[method].push(...runFold(y,fold,ctx,method).target);
  }
  return {y,pilots:P,contexts,expected};
}
function compare(actual,expected,path='result'){
  if(typeof expected==='number'){
    const tol=Number.isInteger(expected)?0:1e-7;
    assert(Number.isFinite(actual)&&Math.abs(actual-expected)<=tol,path+' numerical mismatch');
  }else if(expected!==null&&typeof expected==='object'){
    assert(Object.keys(expected).length===Object.keys(actual).length,path+' shape mismatch');
    for(const k of Object.keys(expected))compare(actual[k],expected[k],path+'.'+k);
  }else assert(actual===expected,path+' mismatch');
}
module.exports={ablate,nativeFixture,compare};
if(require.main===module){
  if(process.argv[2]==='--check'){
    compare(C.selftest(),require('./checks.json'));
    compare(ablate(),require('./ablation.json'));
    console.log('PASS: synthetic profiles, 17-way mixtures, target isolation, golden outputs');
  }else console.log(JSON.stringify(process.argv[2]==='--native-fixture'?nativeFixture():ablate(),null,2));
}
