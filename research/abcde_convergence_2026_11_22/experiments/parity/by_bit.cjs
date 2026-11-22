'use strict';
const fs=require('node:fs'), path=require('node:path'), zlib=require('node:zlib');

function byBit(fixture,trace,includeTargets=false) {
  const groups=new Map(), targets=[];
  const key=(stratum,tau,mode,bit)=>JSON.stringify([stratum,tau,mode,bit]);
  for (let i=0;i<fixture.frames.length;i++) {
    const frame=fixture.frames[i], sample=trace.qam[i];
    if (frame.id!==sample.id || frame.stratum!==sample.stratum) throw Error('unaligned frame');
    for (const {target,tau,mode} of sample.entries) {
      const [row,tone]=target, sym=frame.symbol[row][tone];
      for (let bit=0;bit<4;bit++) {
        const truth=(sym>>(3-bit))&1, orig=mode.original.hard[bit]!==truth;
        for (const [name,rec] of Object.entries(mode)) {
          const id=key(frame.stratum,tau,name,bit);
          if (!groups.has(id)) groups.set(id,{stratum:frame.stratum,tau,mode:name,bit,n:0,errors:0,loss_sum:0,rescues:0,new_errors:0});
          const g=groups.get(id), bad=rec.hard[bit]!==truth, nll=-rec.logpb[bit][truth];
          g.n++; g.errors+=Number(bad); g.loss_sum+=nll;
          g.rescues+=Number(orig&&!bad); g.new_errors+=Number(!orig&&bad);
          if (includeTargets) targets.push({frame:frame.id,stratum:frame.stratum,tau,mode:name,row,tone,bit,truth,hard:rec.hard[bit],nll});
        }
      }
    }
  }
  const positions=[...groups.values()].map(g=>({...g,ber:g.errors/g.n,nll_nats:g.loss_sum/g.n,flips:g.rescues+g.new_errors}));
  return includeTargets?{positions,targets}:{positions};
}
module.exports=byBit;
if (require.main===module) {
  const file=n=>fs.readFileSync(path.join(__dirname,n),'utf8');
  const fixture=JSON.parse(file('fixture.json'));
  const trace=JSON.parse(zlib.gunzipSync(Buffer.from(file('raw.json.gz.b64'),'base64')).toString('utf8'));
  console.log(JSON.stringify(byBit(fixture,trace,process.argv.includes('--per-target')),null,2));
}
