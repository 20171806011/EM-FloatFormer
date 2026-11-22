'use strict';
// Node >= 18, no dependencies. Coordinates are fftshifted, natural-binary 16-QAM.
// Public API fit(y, pilotSCs, fold, nuisance) consumes a single 14x128 grid.
// nuisance, when supplied by the caller, MUST have been built before its target fold.
const S=14, N=128, CP=6, SP=2, K=101, R=17;
const USED=Array.from({length:102},(_,i)=>i+13).filter(k=>k!==64);
const P=[13,21,29,37,45,53,61,69,77,85,93,101,109,114];
const PIL=P.map((_,i)=>[i%4<2?-.7:.7,i%2?.7:-.7]);
const O=[0,-.36,-.27,-.18,-.09,.09,.18,.27,.36];
const J=[-135,-90,-45,45,90,135,180].map(x=>Math.PI*x/180);
const LEVEL=[-3,-1,1,3].map(x=>x/Math.sqrt(10));
const X=LEVEL.flatMap(a=>LEVEL.map(b=>[a,b]));
const zero=()=>[0,0], grid=(n,f)=>Array.from({length:S},(_,s)=>Array.from({length:n},(_,k)=>f(s,k)));
const rot=([a,b],t)=>[a*Math.cos(t)-b*Math.sin(t),a*Math.sin(t)+b*Math.cos(t)];
const mul=([a,b],[c,d])=>[a*c-b*d,a*d+b*c];
const norm=([a,b])=>a*a+b*b;
const data=(s,k)=>s!==SP||!P.includes(USED[k]);
const lse=a=>{const m=Math.max(...a);return m===-Infinity?m:m+Math.log(a.reduce((z,x)=>z+Math.exp(x-m),0))};
const physical=(s,r)=>r<9?O[r]*(s-SP):r<16?(s===SP?0:J[r-9]):0;
const eq=(v,w,t,msg)=>{if(!(Math.abs(v-w)<=t))throw Error(msg+' '+v+' vs '+w)};

function pilotChannel(y,p){
  if(!p.length||p.some((k,i)=>!P.includes(k)||(i&&k<=p[i-1])))throw Error('visible pilots must be sorted subset');
  const hp=p.map(k=>{
    const [a,b]=PIL[P.indexOf(k)], v=y[SP][k];
    return mul(v,[a/(a*a+b*b),-b/(a*a+b*b)]);
  });
  return USED.map(k=>{
    const right=p.findIndex(x=>x>=k);
    if(right<0)return hp.at(-1);
    if(right===0)return hp[0];
    const t=(k-p[right-1])/(p[right]-p[right-1]);
    return hp[right].map((v,i)=>t*v+(1-t)*hp[right-1][i]);
  });
}
function guardVariance(y,h){
  const guards=[0,1,2,3,4,5,122,123,124,125,126,127];
  const power=y.reduce((sum,row)=>sum+guards.reduce((v,k)=>v+norm(row[k]),0),0)/(S*guards.length);
  return Math.max(1e-8, 1e-5*h.reduce((a,x)=>a+norm(x),0)/K,power);
}
// Standalone fallback uses only pilots/guard tones. Supply native fold-safe H_ls,
// H_ca and nv_grid for matched replay. hCA is only needed by the demapper.
function inputs(y,p,fold,nuisance){
  if(![0,1].includes(fold)||y.length!==S||y.some(row=>row.length!==N))throw Error('expected 14x128, fold 0/1');
  const h=nuisance?.hLS??pilotChannel(y,p);
  const v=nuisance?.nv??grid(K,()=>1); 
  if(!nuisance?.nv){const n=guardVariance(y,h);for(const row of v)row.fill(n)}
  const hCA=nuisance?.hCA??h;
  if(h.length!==K||hCA.length!==K||v.length!==S||v.some(row=>row.length!==K||row.some(x=>!(x>0&&Number.isFinite(x)))))throw Error('bad channel/variance dimensions or values');
  return {h,hCA,v,fold};
}
// Independent Gaussian RE likelihood conditional on the frozen pilot estimate and variance.
// The two independent PAM coordinates exactly factor the 16-way QAM mixture.
function evaluator(y,{h,v,fold}){
  const rows=Array.from({length:S},()=>[]);
  for(let s=0;s<S;s++)for(let k=0;k<K;k++){
    if(k%2===fold||!data(s,k)||s===SP)continue;
    const [a,b]=y[s][USED[k]], [c,d]=h[k], n=Math.max(v[s][k],1e-8);
    rows[s].push([(a*c+b*d)/n,(b*c-a*d)/n,(c*c+d*d)/n,-(a*a+b*b)/n-Math.log(16*Math.PI*n)]);
  }
  return function score(om){
    let total=0;
    rows.forEach((row,s)=>{
      const t=(s-SP)*om, co=Math.cos(t),si=Math.sin(t);
      for(const [u,w,e,z] of row){
        const i=u*co+w*si, q=w*co-u*si;
        total+=z+lse(LEVEL.map(a=>2*a*i-e*a*a))+lse(LEVEL.map(a=>2*a*q-e*a*a));
      }
    });
    return total;
  };
}
function fitDiscrete(y,p=P,fold=0,nuisance=null){
  const ctx=inputs(y,p,fold,nuisance), f=evaluator(y,ctx);
  const scores=O.map(f), best=scores.indexOf(Math.max(...scores));
  return {ctx,coarse:O[best],evaluations:9};
}
function refine(score,a,b){
  const rho=(Math.sqrt(5)-1)/2;
  let c=b-rho*(b-a), d=a+rho*(b-a), fc=score(c),fd=score(d);
  for(let j=0;j<30&&b-a>1e-7;j++){
    if(fc>=fd){b=d;d=c;fd=fc;c=b-rho*(b-a);fc=score(c)}
    else{a=c;c=d;fc=fd;d=a+rho*(b-a);fd=score(d)}
  }
  return fc>=fd?c:d;
}
function fit(y,p=P,fold=0,nuisance=null){
  const ctx=inputs(y,p,fold,nuisance), evaluate=evaluator(y,ctx);
  const cache=new Map();
  function score(x){if(!cache.has(x))cache.set(x,evaluate(x));return cache.get(x)}
  const ordered=[...O].sort((a,b)=>a-b);
  const cells=O.map(center=>{
    const i=ordered.indexOf(center);
    const a=i===0?-.40:(center+ordered[i-1])/2;
    const b=i===8?.40:(center+ordered[i+1])/2;
    const raw=[center,...Array.from({length:9},(_,j)=>a+(b-a)*j/8)].sort((x,z)=>x-z);
    const points=raw.filter((x,j)=>j===0||Math.abs(x-raw[j-1])>1e-12);
    let best=center;
    const keep=x=>{if(score(x)>score(best))best=x};
    for(const x of points)keep(x);
    for(let j=0;j<points.length;j++){
      const l=points[Math.max(j-1,0)], r=points[Math.min(j+1,points.length-1)];
      if(score(points[j])>=score(l)&&score(points[j])>=score(r))
        keep(refine(score,l,r));
    }
    return best;
  });
  const best=arr=>arr.reduce((a,b)=>score(a)>=score(b)?a:b);
  return {ctx,omega:best(cells),coarse:best(O),cells,evaluations:cache.size};
}
// Output in native [s,r] order. With shared external pre-target log weights
// both D and C use the identical 17-way mixture. Without weights these are
// single-trajectory profiles for diagnostics (including the synthetic check).
function classicalHead(fitResult,continuous,weights=null){
  const {cells,omega,coarse}=fitResult;
  const winner=continuous?cells.indexOf(omega):O.indexOf(coarse);
  if(weights&&(weights.length!==S||weights.some(row=>row.length!==R||Math.abs(lse(row))>1e-5||!Number.isFinite(lse(row)))))throw Error('weights must be normalized log probabilities');
  return {
    logw:weights??grid(R,(_,r)=>r===winner?0:-Infinity),
    dphi:grid(R,(s,r)=>continuous&&r<9?(cells[r]-O[r])*(s-SP):0),
    lognv:Array(S).fill(0)
  };
}
// Exact native bit log P(1)/P(0), float64 reference, with common variance across r.
// Only the target RE is read at this stage. Each data RE appears in exactly one fold.
function demap(y,{h,hCA,v,fold},head){
  const out=[];
  for(let s=0;s<S;s++)for(let k=fold;k<K;k+=2){
    if(!data(s,k))continue;
    const yy=y[s][USED[k]], n=Math.max(1e-8,v[s][k]*Math.exp(Math.max(-1,Math.min(2,head.lognv[s]))));
    const scores=X.map(x=>{
      const terms=[];
      for(let r=0;r<R;r++){
        const w=head.logw[s][r];
        if(w===-Infinity)continue;
        const z=rot(mul(r===16?hCA[k]:h[k],x),physical(s,r)+head.dphi[s][r]);
        terms.push(w-((yy[0]-z[0])**2+(yy[1]-z[1])**2)/n);
      }
      return lse(terms);
    });
    const llr=[3,2,1,0].map(bit=>lse(scores.filter((_,i)=>(i>>bit)&1))-lse(scores.filter((_,i)=>!((i>>bit)&1))));
    out.push({s,k:USED[k],llr});
  }
  return out;
}
function fixture(om){
  let seed=83;
  const next=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed>>>28};
  const truth=grid(N,()=>next());
  const h=[.89,.31];
  const y=grid(N,(s,k)=>{
    if(!USED.includes(k))return zero();
    const x=s===SP&&P.includes(k)?PIL[P.indexOf(k)]:X[truth[s][k]];
    return rot(mul(h,x),om*(s-SP));
  });
  return {y,truth};
}
// Unitary O(N^2) reference DFT for a small fixture, not part of inference.
function transform(row,sign){
  return Array.from({length:N},(_,k)=>{
    let a=0,b=0;
    for(let m=0;m<N;m++){const z=rot(row[m],sign*2*Math.PI*k*m/N);a+=z[0];b+=z[1]}
    return [a/Math.sqrt(N),b/Math.sqrt(N)];
  });
}
function addCfo(y,eps,noise){
  let state=101;
  const u=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return (state+.5)/4294967296};
  const power=y.reduce((tot,row)=>tot+USED.reduce((v,k)=>v+norm(row[k]),0),0)/(S*K);
  const sd=noise?Math.sqrt(power*10**(-1.6)/2):0;
  const shift=row=>row.slice(N/2).concat(row.slice(0,N/2));
  return y.map((row,s)=>{
    const t=transform(shift(row),1);
    const block=t.slice(-CP).concat(t).map((z,n)=>rot(z,2*Math.PI*eps*((s-SP)*(N+CP)+n-CP-(N-1)/2)/N));
    return shift(transform(block.slice(CP),-1)).map(([a,b])=>{
      const r=Math.sqrt(-2*Math.log(u())), theta=2*Math.PI*u();
      return [a+sd*r*Math.cos(theta),b+sd*r*Math.sin(theta)];
    });
  });
}
function selftest(){
  const results=[];
  for(const om of [0,.2317,-.3143]){
    const {y,truth}=fixture(om);
    let errD=0,errC=0,count=0, estimates=[], coarse=[];
    for(const fold of [0,1]){
      const f=fit(y,P,fold), d=demap(y,f.ctx,classicalHead(f,false)),c=demap(y,f.ctx,classicalHead(f,true));
      eq(f.omega,om,1e-6,'slope');estimates.push(f.omega);coarse.push(f.coarse);
      for(let j=0;j<c.length;j++)for(let bit=0;bit<4;bit++){
        const label=(truth[c[j].s][c[j].k]>>(3-bit))&1;
        errC+=Number((c[j].llr[bit]>0)!==Boolean(label));errD+=Number((d[j].llr[bit]>0)!==Boolean(label));count++;
      }
    }
    eq(count,5600,0,'covered bits');eq(errC,0,0,'continuous bit errors');
    results.push({omega:om,estimate:estimates,discrete:coarse,errD,errC,bits:count});
  }
  const {y}=fixture(.2317),ctx=inputs(y,P,0,null);
  ctx.v=grid(K,()=>.06);
  const score=evaluator(y,ctx), gap=Math.abs(score(.2317)-score(.2317+Math.PI/2));
  eq(gap,0,1e-8,'likelihood alias');
  // Let x' = x exp(-j pi (s-SP)/2). At offset 1 the bits generally change.
  const a=rot(X[0],-Math.PI/2), idx=X.findIndex(x=>Math.hypot(x[0]-a[0],x[1]-a[1])<1e-12);
  eq(idx,3,0,'rotated bits'); 
  const y2=y.map(row=>row.slice());
  for(let s=0;s<S;s++)for(let k=0;k<K;k+=2)if(data(s,k))y2[s][USED[k]]=[11,-23];
  const before=fit(y,P,0),after=fit(y2,P,0);
  eq(before.omega,after.omega,0,'target perturbation');before.cells.forEach((x,i)=>eq(x,after.cells[i],0,'cell perturbation'));
  const eps=.042, omega=2*Math.PI*eps*(N+CP)/N;
  const tone=grid(N,(_,k)=>k===35?[1,0]:zero()), dispersed=addCfo(tone,eps,false);
  const leakage=dispersed[SP].reduce((t,z,k)=>t+(k===35?0:norm(z)),0);
  const predicted=1-(Math.sin(Math.PI*eps)/(N*Math.sin(Math.PI*eps/N)))**2;
  eq(leakage,predicted,1e-12,'ICI');
  const raw=fixture(0), stressed=addCfo(raw.y,eps,true), est=[];
  let errD=0,errC=0;
  for(const fold of [0,1]){
    const cf=fit(stressed,P,fold),df=fitDiscrete(stressed,P,fold);
    const d=demap(stressed,df.ctx,classicalHead(df,false)), c=demap(stressed,cf.ctx,classicalHead(cf,true));
    est.push(cf.omega);eq(cf.omega,omega,.012,'time-domain fixture slope');
    for(let j=0;j<c.length;j++)for(let bit=0;bit<4;bit++){
      const label=Boolean((raw.truth[c[j].s][c[j].k]>>(3-bit))&1);
      errC+=Number((c[j].llr[bit]>0)!==label);errD+=Number((d[j].llr[bit]>0)!==label);
    }
  }
  const fixed=grid(R,()=>-Math.log(R)), ext=classicalHead(before,true,fixed);
  const mixture=demap(y,before.ctx,ext);
  if(mixture.some(re=>re.llr.some(z=>!Number.isFinite(z))))throw Error('17-way mixture');
  // Check a nontrivial mixture against compact y*conj(h) algebra, independent of distance code.
  const jctx=inputs(y,P,0,null);
  jctx.v=grid(K,()=>.08);jctx.hCA=jctx.h.map(z=>rot(z,.23));
  const jw=Array.from({length:R},(_,r)=>-.15*r);
  const normalized=jw.map(v=>v-lse(jw));
  const jhead={logw:grid(R,(_,r)=>normalized[r]),dphi:grid(R,(s,r)=>.19*Math.cos(s+r)),lognv:Array(S).fill(.4)};
  const tested=demap(y,jctx,jhead).find(row=>row.s===7&&row.k===USED[20]);
  const yy=y[7][USED[20]], sc=.08*Math.exp(.4);
  const compact=X.map(x=>lse(jw.map((_,r)=>{
    const h=r===16?jctx.hCA[20]:jctx.h[20], z=mul(yy,[h[0],-h[1]]);
    const u=rot(z,-physical(7,r)-jhead.dphi[7][r]);
    return normalized[r]-(norm(h)*norm(x)-2*(u[0]*x[0]+u[1]*x[1]))/sc;
  })));
  const compactGap=Math.max(...[3,2,1,0].map((bit,b)=>{
    const key=compact.map((z,i)=>(i>>bit)&1);
    const ll=lse(compact.filter((_,i)=>key[i]))-lse(compact.filter((_,i)=>!key[i]));
    return Math.abs(ll-tested.llr[b]);
  }));
  eq(compactGap,0,1e-10,'17-way compact algebra');
  const {y:base}=fixture(.17), baseCtx=inputs(base,P,1,null);baseCtx.v=grid(K,()=>.083);
  const trial=.2381, factor=evaluator(base,baseCtx)(trial);
  let direct=0;
  for(let s=0;s<S;s++)for(let k=0;k<K;k+=2){
    if(s===SP)continue;
    const z=base[s][USED[k]], a=rot(baseCtx.h[k],(s-SP)*trial);
    direct+=lse(X.map(x=>{const m=mul(a,x);return -((m[0]-z[0])**2+(m[1]-z[1])**2)/.083;}))-Math.log(16*Math.PI*.083);
  }
  eq(factor,direct,1e-8,'PAM vs QAM');
  return {checks:results,alias_score_gap:gap,alias_symbols:[0,idx],target_mutation:'pass',
    compact_algebra_gap:compactGap,reference_ll_gap:Math.abs(factor-direct),finite_17way_llrs:mixture.length*4,
    c2_like:{eps,omega,est,errD,errC,bits:5600,ICI_tone_leakage:leakage,ICI_theory:predicted}};
}
const api={fit,fitDiscrete,inputs,evaluator,classicalHead,demap,selftest,fixture,addCfo,
  layout:{S,N,CP,SP,K,R,USED,P,X,data,physical}, math:{lse,mul,rot,norm}};
function headPacket(packet){
  const {y,visiblePilots,fold,nuisance,logw,method}=packet;
  if(!['discrete','continuous'].includes(method)||!logw||!nuisance?.nv||!nuisance.hCA||!nuisance.hLS)
    throw Error('head packet requires method, shared nuisance and pre-target logw');
  const result=method==='continuous'?fit(y,visiblePilots,fold,nuisance):{ctx:inputs(y,visiblePilots,fold,nuisance)};
  const head=classicalHead(result,method==='continuous',logw);
  return {logw:head.logw,dphi:head.dphi,lognv:head.lognv,omega:result.omega??null,evaluations:result.evaluations??0};
}
api.headPacket=headPacket;
if(typeof module!=='undefined'){
  module.exports=api;
  if(typeof require!=='undefined'&&require.main===module){
    if(process.argv[2]==='--head-json'){
      const packet=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
      console.log(JSON.stringify(headPacket(packet)));
    }else console.log(JSON.stringify(selftest(),null,2));
  }
}
