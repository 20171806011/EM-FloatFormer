/* Standalone Node 18+ independent full-grid synthetic calculation. No third-party
   packages. Uses a flat pilot channel and a flat training reference, so h0=h16=1.
   CLI: node fullgrid.js. Coordinates are [OFDM row, active-carrier index].
   audit.py is the general replay interface for arbitrary nonflat reference banks. */
'use strict';
const sum=a=>a.reduce((x,y)=>x+y,0), mean=a=>sum(a)/a.length;
const lse=a=>{const m=Math.max(...a);return m+Math.log(sum(a.map(x=>Math.exp(x-m))))};
const prob=a=>{const m=Math.max(...a),u=a.map(x=>Math.exp(x-m)),z=sum(u);return u.map(x=>x/z)};
const ix=n=>Array.from({length:n},(_,i)=>i);
const clip=(x,n)=>Math.min(n-1,Math.max(0,x));
const pow=([r,i])=>r*r+i*i;
const mul=([r,i],[a,b])=>[r*a-i*b,r*b+i*a];
const conj=([r,i])=>[r,-i], sub=([r,i],[a,b])=>[r-a,i-b];
const angle=a=>[Math.cos(a),Math.sin(a)];
const S=14,K=101,R=17, sy=ix(S),ks=ix(K),rs=ix(R);
const used=ix(128).filter(k=>k>=13&&k<=114&&k!==64);
const pilots=[13,21,29,37,45,53,61,69,77,85,93,101,109,114];
const data=(s,k)=>!(s===2&&pilots.includes(used[k]));
const qam=ix(16).map(i=>[(2*(i>>2)-3)/Math.sqrt(10),(2*(i%4)-3)/Math.sqrt(10)]);
const omega=[0,-.36,-.27,-.18,-.09,.09,.18,.27,.36,...Array(8).fill(0)];
const jump=[...Array(9).fill(0),-135,-90,-45,45,90,135,180,0].map(v=>v*Math.PI/180);
const prior=[.28,...Array(8).fill(.28/8),...Array(7).fill(.28/7),.16].map(Math.log);
const phase=(r,s)=>omega[r]*(s-2)+(s===2?0:jump[r]);
const h=rs.map(r=>sy.map(s=>angle(phase(r,s))));
function synthetic(){
  let state=731;const rand=()=>((state=(1664525*state+1013904223)>>>0)+.5)/4294967296;
  const y=sy.map(s=>ix(128).map(()=>[.15*(2*rand()-1),.15*(2*rand()-1)]));
  for(const s of sy)for(const f of used){
    const x=qam[Math.floor(rand()*16)];y[s][f]=[y[s][f][0]+x[0],y[s][f][1]+x[1]];
  }
  pilots.forEach((f,j)=>y[2][f]=[j%4>=2?.7:-.7,j%2?.7:-.7]);
  y[7][used[45]]=[3.6,2.1];
  return y;
}
function variance(y,keep){
  let guard=[],active=[],res=[];
  for(const s of sy){
    for(const f of [0,1,2,3,4,5,122,123,124,125,126,127])guard.push(pow(y[s][f]));
    for(const k of ks)if(keep(s,k)){
      active.push(pow(y[s][used[k]]));
      if(s===2&&data(s,k))res.push(Math.min(...qam.map(x=>pow(sub(y[s][used[k]],x)))));
    }
  }
  res.sort((a,b)=>a-b);
  const base=Math.min(1.5*mean(active),Math.max(1.25*mean(res.slice(0,Math.floor(.9*res.length))),mean(guard),1e-5));
  const nv=sy.map(()=>[]),excess=sy.map(()=>[]);
  for(const s of sy)for(const k of ks){
    let z=0,n=0;for(let i=-1;i<=1;i++)for(let j=-1;j<=1;j++){
      const a=clip(s+i,S),b=clip(k+j,K);
      if(keep(a,b)){z+=pow(y[a][used[b]]);n++}
    }
    const energy=z/n,ratio=energy/(1+base);
    nv[s][k]=ratio>1.75?Math.min(40*Math.max(base,.5),Math.max(base,energy-1)):base;
    excess[s][k]=Math.log(Math.max(1e-6,ratio));
  }
  return {base,nv,excess};
}
function local(y,hr,v){
  const joint=qam.map(x=>-Math.log(16*Math.PI*v)-pow(sub(y,mul(hr,x)))/v);
  const p=prob(joint),ell=lse(joint);
  const x=[0,1].map(i=>sum(qam.map((q,j)=>q[i]*p[j])));
  const entropy=-sum(p.map(v=>v*Math.log(Math.max(v,1e-300))))/Math.log(16);
  const bit=ix(4).map(b=>sum(p.filter((_,i)=>(i>>(3-b))&1)));
  return {ell,x,entropy,bit,joint};
}
function hmm(score,tau,stay){
  const tr=rs.map(i=>rs.map(j=>Math.log(i===j?stay:(1-stay)/(R-1))));
  const f=[rs.map(r=>prior[r]+score[r][0]/tau)],b=new Array(S);b[S-1]=rs.map(()=>0);
  const center=a=>{const z=lse(a);return a.map(x=>x-z)};
  f[0]=center(f[0]);
  for(let s=1;s<S;s++)f[s]=center(rs.map(j=>score[j][s]/tau+lse(rs.map(i=>f[s-1][i]+tr[i][j]))));
  for(let s=S-2;s>=0;s--)b[s]=center(rs.map(i=>lse(rs.map(j=>tr[i][j]+score[j][s+1]/tau+b[s+1][j]))));
  return sy.map(s=>center(rs.map(r=>f[s][r]+b[s][r])));
}
function fit(y,tar,exclude){
  const [t,k0]=tar,keep=(s,k)=>!(exclude&&s===t&&k===k0);
  const {base,nv,excess}=variance(y,keep);
  const score=rs.map(()=>sy.map(()=>0)),cplx=sy.map(()=>[]),scal=sy.map(()=>[]);
  let target=[];
  for(const r of rs)for(const s of sy){
    let wsum=0,count=0,ent=0,mse=0,zs=ix(3).map(()=>[0,0]);
    for(const k of ks){
      if(!data(s,k))continue;
      const z=y[s][used[k]],hr=h[r][s],v=nv[s][k],u=local(z,hr,v);
      if(s===t&&k===k0)target[r]=u;
      if(!keep(s,k))continue;
      score[r][s]+=u.ell;count++;ent+=u.entropy;
      const eq=mul(z,conj(hr)),res=sub(eq,u.x),w=1/(1+v);
      const amp=Math.max(Math.sqrt(pow(eq)),.25),unit=eq.map(a=>a/amp),square=mul(unit,unit);
      const cs=[mul(eq,conj(u.x)),mul(res,conj(u.x)),mul(square,square)];
      for(let j=0;j<3;j++)for(let dim=0;dim<2;dim++)zs[j][dim]+=w*cs[j][dim];
      wsum+=w;mse+=w*pow(res);
    }
    cplx[s][r]=[...zs.map(z=>z.map(v=>v/wsum)),h[r][s]];
    const diagnostic=mean(ks.filter(k=>keep(s,k)).map(k=>Math.max(0,excess[s][k])))+.1*Math.log1p(1/base);
    scal[s][r]=[0,0,prior[r],1-ent/count,Math.log1p(mse/wsum),(s-2)/11,omega[r]/.4,Math.cos(jump[r]),Math.sin(jump[r]),diagnostic];
  }
  const frame=score.map(mean);
  for(const s of sy)for(const r of rs){
    scal[s][r][0]=(score[r][s]-Math.max(...rs.map(j=>score[j][s])))/100;
    scal[s][r][1]=(frame[r]-Math.max(...frame))/100;
  }
  return {target,base,nv,score,cplx,scal};
}
function mix(lw,tab,normalized=false){
  const logm=ix(16).map(i=>lse(tab.map((u,r)=>lw[r]+u.joint[i]-(normalized?u.ell:0))));
  const bit=ix(4).map(b=>{
    const s1=lse(logm.filter((_,i)=>(i>>(3-b))&1)),s0=lse(logm.filter((_,i)=>!((i>>(3-b))&1)));
    return Math.exp(s1-lse([s1,s0]));
  });
  return {candidate:rs,local_log_marginal:tab.map(z=>z.ell),weight:prob(lw),
          effective_weight:prob(lw.map((v,r)=>v+(normalized?0:tab[r].ell))),branch_p_bit1:tab.map(z=>z.bit),
          p_bit1:bit,hard:bit.map(p=>Number(p>.5))};
}
const delta=(a,b)=>Array.isArray(a)?Math.max(...a.map((x,i)=>delta(x,b[i]))):Math.abs(a-b);
function replay(y,tar,tau=1,stay=.88){
  const [s,k]=tar,raw=fit(y,tar,false),masked=fit(y,tar,true);
  const result={target:{symbol:s,used_index:k,fft_bin:used[k],y:y[s][used[k]]},tau,stay};
  for(const mode of ['original','explicit','rebuild']){
    const b=mode==='rebuild'?masked:raw;
    const sc=b.score.map((e,r)=>e.map((v,u)=>v-(mode==='explicit'&&s===u?b.target[r].ell:0)));
    const lw=hmm(sc,tau,stay)[s];
    result[mode]={...mix(lw,b.target),nv_base:b.base,target_variance:b.nv[s][k],
       row_score:sc.map(x=>x[s]),delta_complex_component_max:delta(b.cplx,raw.cplx),
       delta_scalar_features_max:delta(b.scal,raw.scal)};
    if(mode==='original')result.normalized_full=mix(lw,b.target,true);
  }
  const y2=y.map(row=>row.slice());y2[s][used[k]]=[-.35,.22];
  const altered=fit(y2,tar,true);
  for(const key of ['base','nv','cplx','scal','score'])
    if(delta(masked[key],altered[key])!==0)throw Error('held-out dependence: '+key);
  if(tau===1&&delta(result.explicit.p_bit1,result.normalized_full.p_bit1)>1e-10)throw Error('cavity identity');
  return result;
}
const y=synthetic();
if (typeof process!=='undefined' && process.argv.includes('--grid')) console.log(JSON.stringify(y));
else console.log(JSON.stringify({OFDM:[[7,45],[2,39],[13,100]].map(t=>replay(y,t)),invariance:true}));
