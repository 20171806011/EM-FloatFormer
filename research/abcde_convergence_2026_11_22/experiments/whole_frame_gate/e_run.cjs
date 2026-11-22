'use strict';
// node e_run.cjs fit|score OUT ESTIMATOR_CJS PAIRED_CFO_CJS E_DIRECTORY [LOCK_SHA256]
const fs=require('node:fs'), path=require('node:path'), crypto=require('node:crypto');
const {spawnSync}=require('node:child_process');
const A=require('./e_adapter.cjs');
const pretty=x=>JSON.stringify(x,null,2)+'\n',compact=x=>JSON.stringify(x)+'\n';
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
function main(args){
  const [mode,out0,c0,p0,e0,trusted]=args;
  if(!['fit','score'].includes(mode)||args.length!==(mode==='fit'?5:6))throw Error('fit|score OUT ESTIMATOR PAIRED E_DIR [LOCK_SHA256]');
  const [out,cfile,pfile,eDir]=[out0,c0,p0,e0].map(n=>path.resolve(n));
  const F=n=>path.join(out,n),read=n=>JSON.parse(fs.readFileSync(F(n),'utf8'));
  const put=(n,x,fmt=pretty)=>fs.writeFileSync(F(n),fmt(x),{flag:'wx'});
  fs.mkdirSync(out,{recursive:true});
  const C=require(cfile),E=require(path.join(eDir,'gate.cjs'));
  // paired_cfo.cjs reads process.argv[2] to locate the requested estimator.
  const old=process.argv[2];let P;
  try {process.argv[2]=cfile;P=require(pfile);} finally {process.argv[2]=old;}
  const srcDigests=[require.resolve('./e_adapter.cjs'),pfile,cfile].map(f=>sha(fs.readFileSync(f,'utf8').trimEnd()+'\n'));
  const alg=Object.fromEntries(['frontend','static','dynamic'].map(role=>[role,sha(JSON.stringify(['E-CFO-v1',role,...srcDigests]))]));
  const plan=A.makePlan(P,alg), ecli=path.join(eDir,'cli.cjs');
  const expectedGate='a00576c88a2c54048006dd0da28be0c1b43434a1fdbd602c6e89291edbb2e91f';
  const expectedCli='5785e7ccb851c756562708169892756f7dacab1a69763d4a14c83c5dff5dd569';
  if(sha(fs.readFileSync(path.join(eDir,'gate.cjs')))!==expectedGate||sha(fs.readFileSync(ecli))!==expectedCli)throw Error('E source identity');
  const invoke=(...cliArgs)=>{
    const result=spawnSync(process.execPath,[ecli,...cliArgs],{encoding:'utf8',maxBuffer:32<<20});
    if(result.error || result.status!==0)throw Error(result.stderr||String(result.error));return JSON.parse(result.stdout);
  };
  if(mode==='fit'){
    for(const n of ['plan.json','dev.json','lock.json','dev_report.json','fit_digests.json'])
      if(fs.existsSync(F(n)))throw Error('fit destination exists: '+F(n));
    A.evaluateRows(C,P,E,plan,'fit',null,true);
    const f=A.evaluateRows(C,P,E,plan,'fit',null),s=A.evaluateRows(C,P,E,plan,'select',null);
    const data={alg,rows:[...f.rows,...s.rows]},stats=[...f.stats,...s.stats];
    put('plan.json',plan);put('dev.json',data,compact);
    const dig=invoke('fit',F('plan.json'),F('dev.json'),F('lock.json'));
    put('fit_digests.json',dig);put('dev_report.json',A.reportFit(E,plan,stats,data,read('lock.json').model));
    console.log(JSON.stringify(dig));
  }else{
    if(!/^[0-9a-f]{64}$/.test(trusted)||sha(fs.readFileSync(F('lock.json')))!==trusted)throw Error('precommitted lock digest mismatch before input generation');
    if(pretty(plan)!==fs.readFileSync(F('plan.json'),'utf8'))throw Error('plan/source mismatch');
    const lock=read('lock.json');
    if(lock.plan!==sha(fs.readFileSync(F('plan.json')))||lock.engine!==expectedGate)throw Error('locked provenance mismatch');
    for(const n of ['test.json','engine_score.json','results.json'])if(fs.existsSync(F(n)))throw Error('score destination exists: '+F(n));
    A.evaluateRows(C,P,E,plan,'fit',null,true);
    const v=A.evaluateRows(C,P,E,plan,'test',lock.model),data={alg,rows:v.rows};
    put('test.json',data,compact);
    const score=invoke('score',F('plan.json'),F('lock.json'),trusted,F('test.json'));
    put('engine_score.json',score);
    put('results.json',A.reportTest(E,plan,read('dev_report.json'),v.stats,score));
    console.log(JSON.stringify(score.audit));
  }
}
if(require.main===module)main(process.argv.slice(2));
module.exports={main};
