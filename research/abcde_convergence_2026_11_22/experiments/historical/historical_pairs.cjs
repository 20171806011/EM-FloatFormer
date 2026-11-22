/* Node 18+. Usage: node historical_pairs.cjs <pinned research/realdata_ofdm_stress_and_theory/results directory>.
   Input: two archived per-TTI CSVs, two JSON summaries. No IQ, predictions, or fitting. */
'use strict';
const N=5600, FIRST=502, END=742, SEEDS=[20261121,20261122,20261123];
const BM=['B0_classical_single','B0b_classical_nbi_guard','B1_hard_select','B2a_linear_fusion_within','B2b_linear_fusion_totvar','B3_mixture_likelihood'];
const CM=['RoleA_PhysParallelTransport','RoleB1_MatchedTransformer','RoleB2_MatchedMLP'];
const CONDITIONS=['C0_raw_ota','C1_awgn_stress','C2_cfo_doppler','C3_phase_jump_jitter','C4_nbi_sparse7'];
const ASSERT=(ok,m)=>{if(!ok)throw new Error(m)};
const sum=a=>a.reduce((s,x)=>s+x,0), avg=a=>sum(a)/a.length;
function parse(s,neural) {
  const lines=s.trimEnd().split(/\r?\n/), header=lines.shift().split(',');
  const expected=neural?5*3*3*240:5*6*240;
  ASSERT(lines.length===expected,'row count');
  const out=new Map();
  for(const row of lines){
    const values=row.split(',');ASSERT(values.length===header.length,'column shape');
    const x=Object.fromEntries(header.map((k,i)=>[k,values[i]]));
    const name=neural?x.role:x.method, seed=neural?+x.seed:0, t=+x.tti;
    ASSERT((neural?CM:BM).includes(name)&&CONDITIONS.includes(x.condition),'unexpected series');
    ASSERT((neural?SEEDS:[0]).includes(seed)&&Number.isInteger(t)&&t>=FIRST&&t<END,'unexpected coordinate');
    const key=[name,seed,x.condition,t].join('|');
    ASSERT(!out.has(key),'duplicate row '+key);
    const err=+x.ber*N, nraw=+x.nll_raw,ncal=+x.nll_cal;
    ASSERT(Number.isFinite(err)&&Math.abs(Math.round(err)-err)<1e-8&&err>=0&&err<=N,'invalid BER');
    ASSERT(Number.isFinite(nraw)&&Number.isFinite(ncal),'NLL');
    out.set(key,{err:Math.round(err),ber:+x.ber,nll_raw:nraw,nll_cal:ncal});
  }
  return out;
}
function stats(x) {
  const G=x.length, m=avg(x),se=Math.sqrt(sum(x.map(z=>(z-m)**2))/(G*(G-1)));
  ASSERT(G===12,'fixed block count');
  let above=0;
  for(let mask=0;mask<1<<G;mask++){
    const d=sum(x.map((v,i)=>((mask>>i)&1?1:-1)*v))/G;
    above+=+(Math.abs(d)>=Math.abs(m)-1e-15);
  }
  return {mean:m,ci95_t_df11:[m-2.200985160082949*se,m+2.200985160082949*se],p_two_sided_sign_flip:above/(1<<G),
          blocks_worse_tie_better:[x.filter(v=>v>1e-12).length,x.filter(v=>Math.abs(v)<=1e-12).length,x.filter(v=>v< -1e-12).length]};
}
function analyze(btxt,ctxt,bjson,cjson) {
  const b=parse(btxt,false),c=parse(ctxt,true);
  const get=(name,seed,cond,t)=> (CM.includes(name)?c:b).get([name,CM.includes(name)?seed:0,cond,t].join('|'));
  const report={archive_pin:'5f88a5be984493bcb6aa164d84253b6f29876eef',bit_positions_per_condition:240*N,
    sources:{b_rows:b.size,c_rows:c.size},difference:'candidate minus baseline; positive is more errors or larger loss',
    timings:{b_all_conditions_program_s:bjson.elapsed_s,c_total_program_s:cjson.total_elapsed_s,c_frontend_cache_s:cjson.frontend_cache_time_s,
             c_mean_dev_head_only_ms_per_tti:Object.fromEntries(CM.map(n=>[n,cjson.role_aggregates[n].mean_head_infer_ms_per_tti]))},conditions:{}};
  for(const cond of CONDITIONS){
    const entry={methods:{},pairs:{},b_all_six_end_to_end_ms_per_tti:bjson.conditions[cond].ms_per_tti_all6};
    for(const name of [...BM,...CM]){
      const seeds=CM.includes(name)?SEEDS:[0],rs=seeds.flatMap(seed=>Array.from({length:240},(_,i)=>get(name,seed,cond,FIRST+i)));
      const mm=Object.fromEntries(['ber','nll_raw','nll_cal'].map(k=>[k,avg(rs.map(r=>r[k]))]));
      const reference=CM.includes(name)?cjson.role_aggregates[name].conditions[cond]:bjson.conditions[cond].methods[name];
      for(const k of Object.keys(mm)) ASSERT(Math.abs(mm[k]-reference[k+(CM.includes(name)?'_mean':'')])<1e-10,cond+'/'+name+'/'+k);
      entry.methods[name]=mm;
    }
    // Each edge is an archived system comparison; no claim of a component intervention.
    for(const [a,z] of [[BM[0],BM[5]],[BM[4],BM[5]],[BM[5],CM[0]],[CM[1],CM[0]],[CM[2],CM[0]]]){
      const seeds=(CM.includes(z)||CM.includes(a))?SEEDS:[0];
      const per_seed=[],dt=Array.from({length:240},()=>({ber:0,nll_raw:0,nll_cal:0}));
      for(const seed of seeds){
        let rescuedLo=0,rescuedHi=0,addedLo=0,addedHi=0,deltaErr=0,worse=0,better=0,tie=0;
        for(let t=FIRST;t<END;t++){
          const old=get(a,seed,cond,t),now=get(z,seed,cond,t),e0=old.err,e1=now.err;
          rescuedLo+=Math.max(0,e0-e1);rescuedHi+=Math.min(e0,N-e1);
          addedLo+=Math.max(0,e1-e0);addedHi+=Math.min(e1,N-e0);
          deltaErr+=e1-e0;worse+=+(e1>e0);better+=+(e1<e0);tie+=+(e1===e0);
          for(const k of ['ber','nll_raw','nll_cal'])dt[t-FIRST][k]+=(now[k]-old[k])/seeds.length;
        }
        ASSERT(addedLo-rescuedLo===deltaErr&&addedHi-rescuedHi===deltaErr,'discordance bounds');
        per_seed.push({seed,net_added_minus_rescued:deltaErr,rescued_range:[rescuedLo,rescuedHi],added_range:[addedLo,addedHi],
          tti_worse_tie_better:[worse,tie,better]});
      }
      entry.pairs[a+' -> '+z]={seeds,per_seed,block_stats:Object.fromEntries(['ber','nll_raw','nll_cal'].map(k=>[k,stats(Array.from({length:12},(_,i)=>avg(dt.slice(20*i,20*(i+1)).map(r=>r[k]))))]))};
    }
    report.conditions[cond]=entry;
  }
  return report;
}
if(typeof module!=='undefined' && require.main===module){
  const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
  const hashes={'group_b_per_tti.csv':'eb2c29f3f0407ab5e8ff2b297ca80a004da4cff30fa9117cd8971f3d58442c7c',
                'group_c_per_tti.csv':'4fc0bf43b5d8043add359021c3dc9c2de125b790c1459e9767f275306ee00a17',
                'group_b_results.json':'6518df878bf772ed1f19989e5f1664c7e38cc95b8bc68935b31f563bf3bb8774',
                'group_c_results.json':'6a78923b61af266eeb52a5fc813badb9184832f82f5f413ffbd098ab4caddcf7'};
  const data=Object.entries(hashes).map(([n,digest])=>{const bytes=fs.readFileSync(path.join(process.argv[2],n));
    ASSERT(crypto.createHash('sha256').update(bytes).digest('hex')===digest,'archive SHA256 mismatch: '+n);return bytes.toString('utf8');});
  console.log(JSON.stringify(analyze(data[0],data[1],JSON.parse(data[2]),JSON.parse(data[3])),null,2));
}
