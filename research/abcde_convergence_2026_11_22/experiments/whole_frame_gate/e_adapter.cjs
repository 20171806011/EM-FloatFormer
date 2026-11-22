'use strict';
// Dlab/Clab in reporting correspond to E's static S and dynamic D respectively.
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


const seeds={fit:[61001,61002,61003,61004,61005,61006],select:[62001,62002],test:[63001,63002]};
const layerName=x=>x==='synthetic_static'?'C0_raw_ota':x;
function makePlan(P,alg){
  const conditions=Object.fromEntries(P.scenarios.map(c=>[layerName(c.name),{weight:1/7,primary:'r0'}]));
  return {version:1,raw:'C0_raw_ota',bits:5600,block:1,block_origin:0,guard:0,margin:0,
    margin_basis:'Synthetic static: no positive raw harm budget.',design:'historical',
    evidence:'Synthetic seeds only. C0_raw_ota names the noise-only synthetic static layer for schema compatibility.',
    alg,conditions,ranges:Object.entries(seeds).flatMap(([phase,a])=>a.map(seed=>({phase,session:'syn'+seed,lo:0,hi:0})))};
}
// All fields here precede a decision to launch continuous estimation.
// Shared HMM calculation is the same expression as the paired CFO control.
function front(C,y){
  return [0,1].map(fold=>{const ctx=C.inputs(y,pilots,fold,null);return {ctx,w:sharedGate(y,ctx)}});
}
function zObserved(C,y,F){
  let resid=0,energy=0;
  for(const [i,k] of pilots.entries()){
    const ctx=C.inputs(y,pilots.filter(k2=>k2!==k),0,null),h=ctx.h[used.indexOf(k)];
    const x=[i%4<2?-.7:.7,i%2?.7:-.7],m=product(h,x),v=y[SP][k];
    resid+=(v[0]-m[0])**2+(v[1]-m[1])**2;energy+=v[0]**2+v[1]**2;
  }
  let num=0,den=0;
  for(let s=0;s<S;s++)for(let k=0;k<K;k++)if(data(s,k)){
    const {ctx,w}=F[k%2],ws=w[s].map(Math.exp);
    const hs=ws.map((_,r)=>rotate(r===16?ctx.hCA[k]:ctx.h[k],physical(r,s)));
    const avg=[0,1].map(d=>sum(ws.map((a,r)=>a*hs[r][d])));
    for(let r=0;r<ws.length;r++){
      const [a,b]=hs[r];num+=ws[r]*((a-avg[0])**2+(b-avg[1])**2);den+=ws[r]*(a*a+b*b);
    }
  }
  return {residual:resid/(energy+1e-12),disagreement:num/(den+1e-12)};
}
function receive(C,y,F,dynamic){
  const out=[];let head_ms=0,demap_ms=0;
  for(const {ctx,w} of F){
    const [head,th]=clock(()=>{if(!dynamic)return C.classicalHead({coarse:0},false,w);
      const f=C.fit(y,pilots,ctx.fold,{hLS:ctx.h,hCA:ctx.hCA,nv:ctx.v});
      return C.classicalHead(f,true,w);
    });
    const [part,td]=clock(()=>C.demap(y,ctx,head));head_ms+=th;demap_ms+=td;out.push(...part);
  }
  return {out,head_ms,demap_ms};
}
function loss(truth,ll){
  const signed=truth?ll:-ll;
  assert(Number.isFinite(ll),'invalid LLR');
  return Math.max(-signed,0)+Math.log1p(Math.exp(-Math.abs(signed)));
}
function evaluateRows(C,P,E,plan,phase,model,onlyWarm=false){
  const rows=[],stats=[];
  const hostSeeds=onlyWarm?[65000]:seeds[phase];
  const layers=onlyWarm?P.scenarios.slice(0,1):P.scenarios;
  for(const seed of hostSeeds)for(const sc of layers){
    const {y,label}=P.generate(seed,sc);
    const [F,front_ms]=clock(()=>front(C,y));
    const [z,z_ms]=clock(()=>zObserved(C,y,F));
    let open=null,gate_ms=null,live=null,live_assembly_ms=null;
    if(model){
      [open,gate_ms]=clock(()=>E.decision(model,z)); // before forced C or any continuous output
      [live,live_assembly_ms]=clock(()=>receive(C,y,F,open)); // only the selected head/demapper
    }
    const [s,tS]=clock(()=>receive(C,y,F,false));
    const [d,tD]=clock(()=>receive(C,y,F,true));
    const truth=[],predS=[],predD=[];let nllS=0,nllD=0,nllG=0;
    const stable=(a,b)=>a.s-b.s||a.k-b.k;
    s.out.sort(stable);d.out.sort(stable);if(live)live.out.sort(stable);
    let prev=-1;
    for(let j=0;j<s.out.length;j++){
      const a=s.out[j],b=d.out[j],key=a.s*128+a.k;
      assert(a.s===b.s&&a.k===b.k&&key>prev,'prediction positions');prev=key;
      for(let bit=0;bit<4;bit++){
        const t=(label[a.s][a.k]>>(3-bit))&1, A=a.llr[bit],B=b.llr[bit];
        truth.push(t);predS.push(+(A>0));predD.push(+(B>0));nllS+=loss(t,A);nllD+=loss(t,B);
        if(live){const v=live.out[j];assert(v.s===a.s&&v.k===a.k&&v.llr[bit]===(open?B:A),'live gate routing');nllG+=loss(t,v.llr[bit])}
      }
    }
    assert(truth.length===5600,'wrong bit denominator');
    const session='syn'+seed,condition=layerName(sc.name),sti=truth.join(''),si=predS.join(''),di=predD.join('');
    rows.push({session,tti:0,condition,rep:'r0',z,truth:sti,S:si,D:di});
    let errS=0,errD=0;for(let j=0;j<sti.length;j++){errS+=sti[j]!==si[j];errD+=sti[j]!==di[j]}
    stats.push({phase,session,condition,z,open,bits:5600,errS,errD,nllS,nllD,nllG:live?nllG:null,
      timing:{frontend_ms:front_ms,features_ms:z_ms,gate_ms,forced_static_ms:front_ms+tS,
        forced_dynamic_ms:front_ms+tD,gate_e2e_ms:live?front_ms+z_ms+gate_ms+live_assembly_ms:null,
        selected_head_ms:live?.head_ms??null,selected_demap_ms:live?.demap_ms??null}});
  }
  return {rows,stats};
}
const group=(rs,c)=>rs.filter(r=>r.condition===c);
function reportFit(E,plan,stats,data,m){
  const cs=Object.keys(plan.conditions),sel=stats.filter(r=>r.phase==='select');
  const counts=rs=>({bits:sum(rs.map(z=>z.bits)),eS:sum(rs.map(z=>z.errS)),eD:sum(rs.map(z=>z.errD))});
  const global=counts(sel);
  const ts=[...new Set([null,0,...m.cells.filter(c=>c.enabled&&c.eta>=0).map(c=>c.eta)])];
  const selectRows=data.rows.filter(r=>sel.some(t=>t.session===r.session));
  const thresholds=ts.map(threshold=>{
    const trial={...m,threshold},layer={};
    for(const c of cs){
      const qs=group(selectRows,c).map(r=>E.compare(r,E.decision(trial,r.z)));
      const n=sum(qs.map(z=>z.bits)),eS=sum(qs.map(z=>z.eS)),eG=sum(qs.map(z=>z.eG));
      layer[c]={opens:sum(qs.map(z=>z.opened)),eS,eG,bits:n,benefit:(eS-eG)/n};
    }
    return {threshold,raw_harm:-layer[plan.raw].benefit,
      weighted_gain:sum(cs.map(c=>plan.conditions[c].weight*layer[c].benefit)),layer};
  });
  return {global_select:global,best_fixed:global.eS<=global.eD?'S':'D',model:m,thresholds,
    by_layer:Object.fromEntries(cs.map(c=>[c,counts(group(sel,c))])),dev:stats};
}
function timing(rs){
  const t={};for(const k of Object.keys(rs[0].timing)){const a=rs.map(r=>r.timing[k]).sort((a,b)=>a-b);
    t[k]={median:(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2,p95:a[Math.ceil(a.length*.95)-1]};}
  return t;
}
function reportTest(E,plan,info,stats,score){
  const layers={};
  for(const [c,v] of Object.entries(score.strata)){
    const rs=group(stats,c),n=rs.length*5600,choice=info.best_fixed;
    const extra=(u,h)=>({rescues:u,new_errors:h,flips:u+h,net_added_errors:h-u});
    assert(rs.filter(r=>r.open).length===v.opened&&v.bits===n,'score not aligned with live decision');
    layers[c]={...v,S_to_G:extra(v.uS,v.hS),D_to_G:extra(v.uD,v.hD),
      selected_fixed:choice,fixed_to_G:choice==='S'?extra(v.uS,v.hS):extra(v.uD,v.hD),
      nll:Object.fromEntries(['S','D','G'].map(k=>[k,sum(rs.map(r=>r['nll'+k]))/n])),
      timing_ms:timing(rs)};
  }
  return {layers,weighted:score.weighted,raw:score.raw,inference:score.inference,metrics_by_frame:stats,
    all_eval_timing_ms:timing(stats),score_audit:score.audit};
}
module.exports={seeds,makePlan,evaluateRows,reportFit,reportTest};
