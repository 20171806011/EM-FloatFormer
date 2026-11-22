'use strict';
// node verify_vectors.cjs ARTIFACT_DIR E_DIR NEW_WORK_DIR
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {spawnSync}=require('node:child_process');
const {decode}=require('./vector_codec.cjs');
const pretty=x=>JSON.stringify(x,null,2)+'\n',compact=x=>JSON.stringify(x)+'\n';
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
function main(args){
  if(args.length!==3)throw Error('ARTIFACT_DIR E_DIR NEW_WORK_DIR');
  const [src,edir,out]=args.map(x=>path.resolve(x)), get=n=>fs.readFileSync(path.join(src,n),'utf8'), read=n=>JSON.parse(get(n));
  const at=n=>path.join(out,n);fs.mkdirSync(out,{recursive:false});
  const cli=path.join(edir,'cli.cjs');
  function invoke(...argv){
    const v=spawnSync(process.execPath,[cli,...argv],{encoding:'utf8',maxBuffer:32<<20});
    if(v.error||v.status!==0)throw Error(v.stderr||String(v.error));return JSON.parse(v.stdout);
  }
  function same(x,y,label){if(x!==y)throw Error(label+' mismatch');}
  const manifest=read('manifest.json');
  same(sha(fs.readFileSync(cli)),manifest.e_cli_sha256,'E cli');
  same(sha(fs.readFileSync(path.join(edir,'gate.cjs'))),manifest.e_gate_sha256,'E engine');
  const dig=read('fit_digests.json'),reference=read('engine_score.json');
  const plan=get('plan.json');same(sha(plan),dig.plan_sha256,'plan');fs.writeFileSync(at('plan.json'),plan,{flag:'wx'});
  for(const phase of ['dev','test']){
    const vectors=decode(read(phase+'_vectors.json')),bytes=compact(vectors);
    same(sha(bytes),reference.audit[phase+'_sha256'],phase);
    fs.writeFileSync(at(phase+'.json'),bytes,{flag:'wx'});
  }
  const check=invoke('fit',at('plan.json'),at('dev.json'),at('lock.json'));
  for(const k of ['lock_sha256','dev_sha256','plan_sha256'])same(check[k],dig[k],k);
  same(fs.readFileSync(at('lock.json'),'utf8'),get('lock.json'),'lock bytes');
  const score=invoke('score',at('plan.json'),at('lock.json'),dig.lock_sha256,at('test.json'));
  same(pretty(score),get('engine_score.json'),'whole E score');
  console.log(JSON.stringify({success:true,audit:score.audit,bit_pairs:score.pooled.bits,gate_open_frames:score.pooled.opened}));
}
if(require.main===module)main(process.argv.slice(2));
module.exports={main};
