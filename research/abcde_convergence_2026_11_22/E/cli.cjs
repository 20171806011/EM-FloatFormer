#!/usr/bin/env node
"use strict";
const fs=require("node:fs"), {createHash}=require("node:crypto");
const gate=require("./gate.cjs");
const sha=b=>createHash("sha256").update(b).digest("hex");
const read=file=>{const bytes=fs.readFileSync(file);return {hash:sha(bytes),value:JSON.parse(bytes.toString("utf8"))};};
const out=x=>JSON.stringify(x,null,2)+"\n";
function main(args) {
  if (args.length===4 && args[0]==="fit") {
    const [,planFile,devFile,lockFile]=args, p=read(planFile), d=read(devFile);
    const lock={version:1,plan:p.hash,dev:d.hash,engine:sha(fs.readFileSync(require.resolve("./gate.cjs"))),model:gate.fit(p.value,d.value)};
    const bytes=out(lock);
    fs.writeFileSync(lockFile,bytes,{flag:"wx"});
    return {lock_sha256:sha(bytes),plan_sha256:p.hash,dev_sha256:d.hash};
  }
  if (args.length===5 && args[0]==="score") {
    const [,planFile,lockFile,trustedDigest,testFile]=args;
    const p=read(planFile),l=read(lockFile);
    if (!/^[0-9a-f]{64}$/.test(trustedDigest) || l.hash!==trustedDigest) throw Error("lock digest mismatch");
    if (l.value.version!==1 || l.value.plan!==p.hash || l.value.engine!==sha(fs.readFileSync(require.resolve("./gate.cjs")))) throw Error("plan or engine differs from lock");
    const t=read(testFile);
    return {audit:{plan_sha256:p.hash,lock_sha256:l.hash,dev_sha256:l.value.dev,test_sha256:t.hash,engine_sha256:l.value.engine},...gate.score(p.value,l.value.model,t.value)};
  }
  throw Error("fit PLAN DEV NEW_LOCK | score PLAN LOCK PRECOMMITTED_LOCK_SHA256 TEST");
}
try { process.stdout.write(out(main(process.argv.slice(2)))); }
catch(e) { process.stderr.write(e.message+"\n"); process.exitCode=1; }
