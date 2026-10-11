(function(){
"use strict";
const $=id=>document.getElementById(id), stages=[...document.querySelectorAll(".stage")];
const ui={input:$("urlInput"),start:$("startButton"),status:$("loadingStatus"),present:$("presentButton"),raw:$("stage0Result"),canonical:$("stage1Result"),chars:$("stage2Result"),features:$("featureGrid"),fusion:$("stage4Result"),seeds:[$("seed42"),$("seed43"),$("seed44")],weighted:[$("weighted42"),$("weighted43"),$("weighted44")],biases:[$("bias42"),$("bias43"),$("bias44")],logits:[$("logit42"),$("logit43"),$("logit44")],mean:$("meanScore"),fill:$("gaugeFill"),marker:$("gaugeMarker"),policy:$("stage6Result"),decision:$("finalDecision"),math:{raw:$("equationRaw"),canonical:$("equationCanonical"),chars:$("equationChars"),lexical:$("equationLexical"),fusion:$("equationFusion"),scores:$("equationScores"),threshold:$("equationThreshold"),action:$("equationAction")},structure:{scheme:$("structureScheme"),subdomain:$("structureSubdomain"),domain:$("structureDomain"),suffix:$("structureSuffix"),port:$("structurePort"),path:$("structurePath"),query:$("structureQuery"),fragment:$("structureFragment"),example:$("structureExample")}};
let data=null,sessions=[],prep=null,runId=0;
async function json(path){const r=await fetch(path);if(!r.ok)throw Error("Unable to load "+path);return r.json()}
async function init(){
 if(data)return;
 if(location.protocol==="file:")throw Error("Run start_server.sh first, then open http://127.0.0.1:8765");
 ui.status.textContent="Loading models";
 ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL("vendor/",location.href).href;
 const x=await Promise.all(["bundle_manifest","vocabulary","scaler","psl_rules","html_entities","policy","classifier_biases"].map(n=>json("models/"+n+".json")));
 data={manifest:x[0],vocabulary:x[1],scaler:x[2],pslRules:x[3],htmlEntities:x[4],policy:x[5],classifierBiases:x[6].bias_by_seed};
 prep=PhishingPreprocessing.createPreprocessor(data);
 sessions=await Promise.all(data.manifest.models.map(m=>ort.InferenceSession.create("models/"+m.filename,{executionProviders:["wasm"],graphOptimizationLevel:"all"})));
 ui.status.textContent="All three models loaded.";ui.status.className="loading-status ready";
}
function reset(){
 stages.forEach(e=>e.classList.remove("active","done"));ui.raw.textContent="Waiting for a URL.";ui.canonical.textContent="The normalised URL will appear here.";ui.chars.textContent="Character preview.";ui.features.innerHTML="<span>Feature values will appear here.</span>";ui.fusion.textContent="Waiting for both branches.";[...ui.seeds,...ui.weighted,...ui.biases,...ui.logits].forEach(e=>e.textContent="—");ui.mean.textContent="—";ui.fill.style.width="0%";ui.marker.style.left="0%";ui.policy.textContent="No policy result yet.";ui.decision.className="decision neutral";ui.decision.querySelector(".decision-icon").textContent="?";ui.decision.querySelector("strong").textContent="Waiting for analysis";ui.decision.querySelector("p").textContent="The final model-only tier will appear here.";
 Object.values(ui.math).forEach(element=>{element.textContent="Numbers will appear during this step.";element.classList.remove("ready")});
 Object.entries(ui.structure).forEach(([name,element])=>{if(name!=="example")element.textContent="—"});
 ui.structure.example.textContent="Enter a URL and analyze it to divide it into parts.";
}
function rawFeature(input,name){const i=data.scaler.feature_names.indexOf(name);return i<0?0:Number(input.rawLexical[i])}
const fixed=(n,d=3)=>Number(n).toFixed(d);
const signed=n=>(n>=0?"+":"")+fixed(n,4);
function showMath(element,text){element.textContent=text;element.classList.add("ready")}
function showUrlStructure(input){
 const parsed=new URL(input.scheme+"://"+input.canonicalUrl),parts=prep.extractPsl(input.hostname);
 const values={scheme:input.scheme,subdomain:parts.subdomain||"(none)",domain:parts.domain||input.hostname,suffix:parts.suffix||"(none)",port:parsed.port||"(default)",path:parsed.pathname&&parsed.pathname!=="/"?parsed.pathname:"(none)",query:parsed.search?parsed.search.slice(1):"(none)",fragment:parsed.hash?parsed.hash.slice(1):"(none)"};
 Object.entries(values).forEach(([name,value])=>{ui.structure[name].textContent=value});
 ui.structure.example.textContent="Hostname = ["+[values.subdomain,values.domain,values.suffix].filter(value=>value!=="(none)").join(".")+"]";
}
function showFeatures(input){
 const f=(n)=>rawFeature(input,n);
 const values=[["URL length",Math.round(Math.expm1(f("log_url_length")))],["Hostname length",Math.round(Math.expm1(f("log_hostname_length")))],["Digit ratio",fixed(f("url_digit_ratio"))],["Symbol ratio",fixed(f("url_special_ratio"))],["Subdomains",Math.round(f("subdomain_label_count"))],["Path depth",Math.round(f("path_depth"))],["Query parameters",Math.round(f("query_parameter_count"))],["Hostname entropy",fixed(f("hostname_entropy"))],["IP hostname",f("ip_hostname")>=.5?"yes":"no"],["Public suffix",f("valid_public_suffix")>=.5?"yes":"no"]];
 ui.features.textContent="";values.forEach(v=>{const box=document.createElement("div"),a=document.createElement("span"),b=document.createElement("strong");box.className="feature";a.textContent=v[0];b.textContent=String(v[1]);box.append(a,b);ui.features.appendChild(box)});
}
function tier(score){return data.policy.tiers.find(t=>(t.lower_inclusive===null||score>=t.lower_inclusive)&&(t.upper_exclusive===null||score<t.upper_exclusive))}
async function score(input){
 const tokens=new ort.Tensor("int64",input.tokens,[1,128]),lexical=new ort.Tensor("float32",input.lexical,[1,35]),out=[];
 for(const session of sessions){const r=await session.run({tokens,lexical});out.push({probability:Number(r.probability.data[0]),logit:Number(r.logit.data[0])})}return out;
}
const descriptions={allow:["✓","Allow normally","No strong URL-text warning was found."],secondary_review:["i","Secondary review","Unusual evidence is present; seek independent evidence without blocking."],caution:["!","Caution","The recall-oriented threshold was crossed; show a passive notice."],warning:["!","Warning","Strong URL-text evidence was found; show a prominent warning."],block:["×","Block tier","The strict boundary was crossed; Normal mode may pause navigation with an override."]};
async function run(){
 if(ui.start.disabled)return;
 const id=++runId;reset();ui.start.disabled=true;
 try{
  await init();
  const entered=ui.input.value.trim();if(!entered)throw Error("Please enter a URL.");
  ui.status.textContent="Analyzing URL…";ui.status.className="loading-status";
  const input=prep.prepare(entered),outputs=await score(input);
  if(id!==runId)return;

  showUrlStructure(input);
  ui.raw.textContent=entered;showMath(ui.math.raw,"|u| = "+Array.from(entered).length+" characters");
  ui.canonical.textContent=input.canonicalUrl;showMath(ui.math.canonical,"C(u) = "+input.canonicalUrl+"   |C(u)| = "+Array.from(input.canonicalUrl).length);
  const chars=Array.from(input.canonicalUrl),used=Math.min(chars.length,127),padding=128-used-1,preview=chars.slice(0,127).join("");
  ui.chars.textContent=preview+(chars.length>127?" … [truncated for character branch]":"")+" ("+used+" character tokens + EOS)";
  showMath(ui.math.chars,"128 tokens = "+used+" URL characters + 1 EOS + "+padding+" PAD; CNN output = 128 channels × 3 widths = 384 values");
  showFeatures(input);
  const lengthIndex=data.scaler.feature_names.indexOf("log_url_length"),rawLength=input.rawLexical[lengthIndex],center=data.scaler.center[lengthIndex],scale=data.scaler.scale[lengthIndex]||1,scaled=input.lexical[lengthIndex];
  showMath(ui.math.lexical,"q_length = ("+fixed(rawLength,4)+" − "+fixed(center,4)+") / "+fixed(scale,4)+" = "+fixed(scaled,4)+"; 35 scaled inputs → 32 learned values");
  ui.fusion.textContent="384 CNN values + 32 lexical values → 416 fused values → 128-unit fusion layer.";
  showMath(ui.math.fusion,["Concatenation: 384 + 32 = 416 input values","Matrix operation: (128 × 416) · (416 × 1) + (128 × 1) = (128 × 1)","Fusion parameters: (128 × 416) + 128 = 53,376","ReLU output: h contains 128 non-negative values"].join("\n"));
  const scores=outputs.map(item=>item.probability),logits=outputs.map(item=>item.logit),biases=data.manifest.models.map(record=>Number(data.classifierBiases[String(record.seed)])),weighted=logits.map((value,index)=>value-biases[index]);
  scores.forEach((value,index)=>ui.seeds[index].textContent=fixed(value,6));weighted.forEach((value,index)=>ui.weighted[index].textContent=signed(value));biases.forEach((value,index)=>ui.biases[index].textContent=signed(value));logits.forEach((value,index)=>ui.logits[index].textContent=signed(value));
  const mean=scores.reduce((a,b)=>a+b,0)/scores.length;ui.mean.textContent=fixed(mean,6);
  showMath(ui.math.scores,["Seed 42: "+signed(weighted[0])+" + ("+signed(biases[0])+") = "+signed(logits[0])+" → σ(ℓ₄₂) = "+fixed(scores[0],6),"Seed 43: "+signed(weighted[1])+" + ("+signed(biases[1])+") = "+signed(logits[1])+" → σ(ℓ₄₃) = "+fixed(scores[1],6),"Seed 44: "+signed(weighted[2])+" + ("+signed(biases[2])+") = "+signed(logits[2])+" → σ(ℓ₄₄) = "+fixed(scores[2],6),"Ensemble: ("+scores.map(value=>fixed(value,6)).join(" + ")+") / 3 = "+fixed(mean,6)].join("\n"));
  const result=tier(mean),threshold=Number(data.policy.primary_binary_rule.threshold),binary=mean>=threshold?"phishing":"legitimate",percent=Math.max(0,Math.min(100,mean*100));
  ui.fill.style.width=percent+"%";ui.marker.style.left="calc("+percent+"% - 2px)";
  ui.policy.textContent="Mean score "+fixed(mean,6)+" falls in the “"+result.name.replace("_"," ")+"” band. Primary phishing threshold: "+fixed(threshold,6)+".";
  showMath(ui.math.threshold,fixed(mean,6)+(mean>=threshold?" ≥ ":" < ")+fixed(threshold,6)+" → binary result: "+binary+"; policy tier: "+result.name.replace("_"," "));
  const copy=descriptions[result.name]||["?",result.name,""];
  ui.decision.className="decision "+result.name;ui.decision.querySelector(".decision-icon").textContent=copy[0];ui.decision.querySelector("strong").textContent=copy[1];ui.decision.querySelector("p").textContent=copy[2];
  showMath(ui.math.action,"Input tier = "+result.name.replace("_"," ")+" → displayed action = "+copy[1]);
  stages.forEach(e=>{e.classList.remove("active");e.classList.add("done")});
  ui.status.textContent="Analysis complete.";ui.status.className="loading-status ready";
 }catch(e){ui.status.textContent=e.message||String(e);ui.status.className="loading-status error"}finally{if(id===runId)ui.start.disabled=false}
}
ui.start.addEventListener("click",run);ui.input.addEventListener("keydown",e=>{if(e.key==="Enter")run()});
document.querySelectorAll("[data-example]").forEach(b=>b.addEventListener("click",()=>{ui.input.value=b.dataset.example;ui.input.focus()}));
ui.present.addEventListener("click",()=>{document.body.classList.toggle("present");if(document.body.classList.contains("present")&&document.documentElement.requestFullscreen)document.documentElement.requestFullscreen().catch(()=>{});else if(document.fullscreenElement)document.exitFullscreen().catch(()=>{})});
const query=new URLSearchParams(location.search),requestedUrl=query.get("url");if(requestedUrl)ui.input.value=requestedUrl;
init().then(()=>{ui.start.disabled=false;if(query.get("autoplay")==="1")run()}).catch(e=>{ui.status.textContent=e.message||String(e);ui.status.className="loading-status error"});
})();
