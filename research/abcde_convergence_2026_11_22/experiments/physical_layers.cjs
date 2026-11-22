'use strict';
// Use the exact synthetic frame generator used by the CFO experiment.
const {generate,scenarios}=require('./cfo/paired_cfo.cjs');
const plan=require('./physical_plan.json');
const K=101,S=14,R=17,M=16, fft=Array.from({length:102},(_,i)=>13+i).filter(x=>x!==64);
const pilots=[13,21,29,37,45,53,61,69,77,85,93,101,109,114];
const pilotK=pilots.map(f=>fft.indexOf(f));
const keep=(s,k)=>!(s===2&&pilotK.includes(k));
const x=Array.from({length:16},(_,m)=>[(2*(m>>2)-3)/Math.sqrt(10),(2*(m%4)-3)/Math.sqrt(10)]);
const prior=[.28,...Array(8).fill(.28/8),...Array(7).fill(.28/7),.16].map(Math.log);
const om=[0,-.36,-.27,-.18,-.09,.09,.18,.27,.36];
const jumps=[-135,-90,-45,45,90,135,180].map(x=>x*Math.PI/180);
const sum=a=>a.reduce((v,z)=>v+z,0), avg=a=>sum(a)/a.length;
const abs2=([a,b])=>a*a+b*b;
const clip=(n,max)=>Math.max(0,Math.min(max-1,n));
const lse=a=>{const m=Math.max(...a);return m+Math.log(sum(a.map(v=>Math.exp(v-m))))};
const norm=a=>{const z=lse(a);return a.map(v=>v-z)};
const coords=Array.from({length:S*K},(_,i)=>[Math.floor(i/K),i%K]).filter(([s,k])=>keep(s,k));
function pilotsOnly(y){
  const hp=pilots.map((f,i)=>{
    const [a,b]=y[2][f],c=i%4<2?-.7:.7,d=i%2?.7:-.7,den=c*c+d*d;
    return [(a*c+b*d)/den,(b*c-a*d)/den];
  });
  const flat=[avg(hp.map(z=>z[0])),avg(hp.map(z=>z[1]))];
  const ls=fft.map(f=>{
    const i=pilots.findIndex(k=>k>=f);
    if(!i||pilots[i]===f)return hp[i];
    const a=(f-pilots[i-1])/(pilots[i]-pilots[i-1]);
    return [0,1].map(j=>(1-a)*hp[i-1][j]+a*hp[i][j]);
  });
  const ca=ls.map(z=>z.map((v,j)=>.45*v+.55*flat[j]));
  const h=Array.from({length:R},(_,r)=>Array.from({length:S},(_,s)=>{
    const phi=r<9?om[r]*(s-2):r===16||s===2?0:jumps[r-9];
    const c=Math.cos(phi),d=Math.sin(phi);
    return (r===16?ca:ls).map(([a,b])=>[a*c-b*d,a*d+b*c]);
  }));
  return {hp,ls,h};
}
function fitVariance(y,{hp,ls},omit){
  const valid=(s,k)=>!omit||s!==omit[0]||k!==omit[1];
  const power=ls.map(abs2),guard=[],active=[],res=[];
  for(let s=0;s<S;s++){
    for(const f of [0,1,2,3,4,5,122,123,124,125,126,127])guard.push(abs2(y[s][f]));
    for(let k=0;k<K;k++)if(valid(s,k)){
      const z=y[s][fft[k]];active.push(abs2(z));
      if(s===2&&keep(s,k)){
        const [a,b]=ls[k];
        res.push(Math.min(...x.map(([c,d])=>(z[0]-a*c+b*d)**2+(z[1]-a*d-b*c)**2)));
      }
    }
  }
  res.sort((a,b)=>a-b);
  const pd=hp.slice(2).map((z,i)=>abs2(z.map((q,j)=>q-2*hp[i+1][j]+hp[i][j])));
  const base0=Math.max(1.25*avg(res.slice(0,Math.floor(.9*res.length))),avg(guard),.35*.98*avg(pd)/6,1e-5*Math.max(avg(power),1e-6));
  const base=avg(active)>0?Math.min(1.5*avg(active),base0):base0;
  const median=power.slice().sort((a,b)=>a-b)[50];
  return Array.from({length:S},(_,s)=>Array.from({length:K},(_,k)=>{
    let e=0,n=0;
    for(let i=-1;i<=1;i++)for(let j=-1;j<=1;j++){
      const a=clip(s+i,S),b=clip(k+j,K);
      if(valid(a,b)){e+=abs2(y[a][fft[b]]);n++;}
    }
    e/=n;
    return e/Math.max(power[k]+base,1e-12)>1.75
      ? Math.min(40*Math.max(base,.5*median),Math.max(base,e-power[k])):base;
  }));
}
function joint(z,h,v,out){
  const [a,b]=h,den=Math.max(v,1e-12),constant=-Math.log(16*Math.PI*den);
  let max=-Infinity;
  for(let m=0;m<M;m++){const [c,d]=x[m];out[m]=constant-((z[0]-a*c+b*d)**2+(z[1]-a*d-b*c)**2)/den;max=Math.max(max,out[m]);}
  let total=0;for(let m=0;m<M;m++)total+=Math.exp(out[m]-max);
  return max+Math.log(total);
}
function smooth(score){
  const tr=prior.map((_,i)=>prior.map((_,j)=>Math.log(i===j?plan.stay:(1-plan.stay)/(R-1))));
  const f=[norm(prior.map((w,i)=>w+score[0][i]/plan.tau))],b=Array(S);b[S-1]=prior.map(()=>0);
  for(let s=1;s<S;s++)f[s]=norm(prior.map((_,j)=>score[s][j]/plan.tau+lse(prior.map((_,i)=>f[s-1][i]+tr[i][j]))));
  for(let s=S-2;s>=0;s--)b[s]=norm(prior.map((_,i)=>lse(prior.map((_,j)=>tr[i][j]+score[s+1][j]/plan.tau+b[s+1][j]))));
  return f.map((v,s)=>norm(v.map((w,r)=>w+b[s][r])));
}
function fit(y,h,omit){
  const v=fitVariance(y,h,omit),score=Array.from({length:S},()=>new Float64Array(R)),out=new Float64Array(16);
  for(const [s,k] of coords)if(!omit||s!==omit[0]||k!==omit[1]){
    for(let r=0;r<R;r++)score[s][r]+=joint(y[s][fft[k]],h.h[r][s][k],v[s][k],out);
  }
  return {v,w:smooth(score)};
}
function read(y,h,state,[s,k],remove){
  const logv=prior.map(()=>new Float64Array(16));
  const ell=prior.map((_,r)=>joint(y[s][fft[k]],h.h[r][s][k],state.v[s][k],logv[r]));
  const mass=x.map((_,m)=>lse(prior.map((_,r)=>state.w[s][r]+logv[r][m]-(remove?ell[r]/plan.tau:0))));
  return [0,1,2,3].map(b=>lse(mass.filter((_,m)=>(m>>(3-b))&1))-lse(mass.filter((_,m)=>!((m>>(3-b))&1))));
}
function decode(y,target,mode){
  const h=pilotsOnly(y);
  const common=mode==='strict'?null:fit(y,h,null);
  return target.map(sk=>read(y,h,common||fit(y,h,sk),sk,mode==='explicit'));
}
function count(sink,truth,ll,original){
  ll.forEach((a,i)=>a.forEach((v,b)=>{
    const bit=(truth[i]>>(3-b))&1,wrong=+(Number(v>0)!==bit),priorBad=+(Number(original[i][b]>0)!==bit);
    const z=(1-2*bit)*v;
    sink[b].n++;sink[b].errors+=wrong;
    sink[b].rescues+=priorBad*(1-wrong);sink[b].new_errors+=(1-priorBad)*wrong;
    sink[b].nll_sum+=Math.max(0,z)+Math.log1p(Math.exp(-Math.abs(z)));
  }));
}
const empty=()=>[0,1,2,3].map(()=>({n:0,errors:0,nll_sum:0,rescues:0,new_errors:0}));
function report(a){
  const reduce=arr=>arr.reduce((t,v)=>{for(const k in v)t[k]=(t[k]||0)+v[k];return t},{});
  const annotate=v=>({...v,ber:v.errors/v.n,nll_nats:v.nll_sum/v.n,flips:v.rescues+v.new_errors});
  return {by_bit:a.map(annotate),total:annotate(reduce(a))};
}
function run(){
  const results=[];
  for(const condition of plan.conditions){
    const scenario=scenarios.find(z=>z.name===condition);
    for(const scope of ['same_targets','whole']){
      const selected=scope==='whole'?coords:plan.targets;
      const methods=scope==='whole'?['original','explicit']:['original','explicit','strict'];
      const accum=Object.fromEntries(methods.map(m=>[m,empty()])),times=Object.fromEntries(methods.map(m=>[m,[]]));
      const perFrame=[];
      for(const seed of plan.test_seeds){
        const {y,label}=generate(seed,scenario);
        const truth=selected.map(([s,k])=>label[s][fft[k]]);
        let orig;const frame={seed,errors:{}};
        for(const mode of methods){
          const start=Date.now(),ll=decode(y,selected,mode);times[mode].push(Date.now()-start);
          if(mode==='original')orig=ll;
          if(ll.some(row=>row.some(v=>!Number.isFinite(v))))throw Error('nonfinite LLR');
          count(accum[mode],truth,ll,orig);
          const per=empty();count(per,truth,ll,orig);frame.errors[mode]=report(per).total.errors;
        }
        perFrame.push(frame);
      }
      const stats=Object.fromEntries(methods.map(m=>[m,report(accum[m])]));
      for(const m of methods){const u=stats[m].total;
        if(u.errors-stats.original.total.errors!==u.new_errors-u.rescues)throw Error('paired mismatch');
      }
      results.push({condition,scope,stats,frame_ms:times,paired_frames:perFrame});
    }
  }
  return results;
}
module.exports={run,fit,read,pilotsOnly,decode};
if(require.main===module)console.log(JSON.stringify(run(),null,2));
