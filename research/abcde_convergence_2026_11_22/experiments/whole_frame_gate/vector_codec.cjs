'use strict';
// Compact, lossless archive of generated synthetic labels, both fixed decisions, and E features.
// Bits are packed MSB first in each byte. Session truth is stored once for all seven conditions.
const symbols='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function pack(bits){
  if(bits.length%8 || /[^01]/.test(bits))throw Error('expected a whole number of binary octets');
  const bytes=Array.from({length:bits.length/8},(_,i)=>parseInt(bits.slice(i*8,i*8+8),2));
  let out='';
  for(let i=0;i<bytes.length;i+=3){
    const x=(bytes[i]<<16)|((bytes[i+1]??0)<<8)|(bytes[i+2]??0);
    out+=symbols[(x>>>18)&63]+symbols[(x>>>12)&63]+(i+1<bytes.length?symbols[(x>>>6)&63]:'=')+(i+2<bytes.length?symbols[x&63]:'=');
  }
  return out;
}
function unpack(s){
  if(s.length%4||!/^[A-Za-z0-9+/]*={0,2}$/.test(s))throw Error('bad base64');
  let bits='';
  const toBits=n=>n.toString(2).padStart(8,'0');
  for(let i=0;i<s.length;i+=4){
    const [a,b,c,d]=Array.from(s.slice(i,i+4),k=>k==='='?0:symbols.indexOf(k));
    const x=(a<<18)|(b<<12)|(c<<6)|d;
    bits+=toBits((x>>>16)&255);if(s[i+2]!=='=')bits+=toBits((x>>>8)&255);if(s[i+3]!=='=')bits+=toBits(x&255);
  }
  return bits;
}
function encode(data){
  const hostTruth={},rows=[];
  for(const x of data.rows){
    const host=JSON.stringify([x.session,x.tti]),truth=pack(x.truth);
    if(host in hostTruth&&hostTruth[host]!==truth)throw Error('host label mismatch');hostTruth[host]=truth;
    rows.push({session:x.session,tti:x.tti,condition:x.condition,rep:x.rep,z:x.z,S:pack(x.S),D:pack(x.D)});
  }
  return {version:1,encoding:'base64-msb-first',alg:data.alg,hostTruth,rows};
}
function decode(compact){
  if(compact.version!==1||compact.encoding!=='base64-msb-first')throw Error('unknown format');
  return {alg:compact.alg,rows:compact.rows.map(x=>({session:x.session,tti:x.tti,condition:x.condition,rep:x.rep,z:x.z,
    truth:unpack(compact.hostTruth[JSON.stringify([x.session,x.tti])]),S:unpack(x.S),D:unpack(x.D)}))};
}
if(require.main===module){
  const [mode,input,output]=process.argv.slice(2);
  if(!['encode','decode'].includes(mode)||!input||!output)throw Error('encode|decode INPUT OUTPUT');
  const fs=require('node:fs'),x=JSON.parse(fs.readFileSync(input,'utf8')),value=mode==='encode'?encode(x):decode(x);
  fs.writeFileSync(output,JSON.stringify(value)+'\n',{flag:'wx'});
}
module.exports={encode,decode};
