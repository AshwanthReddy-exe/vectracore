import sys

with open('src/main/resources/index.html', 'r') as f:
    content = f.read()

# 1. ADD CSS
css_to_add = """  .scatter-panel { flex:1; min-width:380px; position:relative; overflow:hidden; border-right:1px solid var(--border); background:var(--bg); }
  #scatter { display:block; width:100%; height:100%; }"""

content = content.replace("/* CENTER — LIBRARY + SEARCH */", css_to_add + "\n\n/* CENTER — LIBRARY + SEARCH */")

# 2. ADD HTML
html_to_add = """  <!-- SCATTER PANEL -->
  <div class="scatter-panel">
    <canvas id="scatter"></canvas>
  </div>
"""

content = content.replace("  <!-- CENTER — LIBRARY + SEARCH TABS -->", html_to_add + "\n  <!-- CENTER — LIBRARY + SEARCH TABS -->")


# 3. Add JS state variables
js_state = """let allItems = [], pcaPoints = [], hitIds = new Set(), queryPt = null;
let hoverItem = null, pulse = 0;
let bounds={minX:-1,maxX:1,minY:-1,maxY:1};
"""

content = content.replace("let selAlgo = 'hnsw', searchResults = [];", "let selAlgo = 'hnsw', searchResults = [];\n" + js_state)

# 4. Add PCA and canvas functions
pca_funcs = """
// PCA AND SCATTER PLOT
const sc = document.getElementById('scatter'), ctx = sc.getContext('2d');

function pca2D(embs) {
  const n = embs.length, d = embs[0].length;
  if (n < 2) return embs.map(() => [0,0]);
  const mean = new Array(d).fill(0);
  for (const e of embs) for (let i=0;i<d;i++) mean[i]+=e[i]/n;
  const X = embs.map(e => e.map((v,i)=>v-mean[i]));
  function powerIter(X,excl) {
    let v = new Array(d).fill(0).map(()=>Math.random()-.5);
    if (excl) { let dot=v.reduce((s,vi,i)=>s+vi*excl[i],0); v=v.map((vi,i)=>vi-dot*excl[i]); }
    let nrm = Math.sqrt(v.reduce((s,vi)=>s+vi*vi,0));
    v = v.map(vi=>vi/nrm);
    for (let it=0;it<200;it++) {
      const Xv=X.map(xi=>xi.reduce((s,xij,j)=>s+xij*v[j],0));
      const nv=new Array(d).fill(0);
      for (let k=0;k<n;k++) for (let j=0;j<d;j++) nv[j]+=X[k][j]*Xv[k];
      if (excl) { let dot=nv.reduce((s,vi,i)=>s+vi*excl[i],0); for (let i=0;i<d;i++) nv[i]-=dot*excl[i]; }
      nrm=Math.sqrt(nv.reduce((s,vi)=>s+vi*vi,0));
      if (nrm<1e-10) break;
      const prev=v.slice(); v=nv.map(vi=>vi/nrm);
      if (v.reduce((s,vi,i)=>s+(vi-prev[i])**2,0)<1e-12) break;
    }
    return v;
  }
  const pc1=powerIter(X,null), pc2=powerIter(X,pc1);
  return X.map(x=>[x.reduce((s,v,i)=>s+v*pc1[i],0),x.reduce((s,v,i)=>s+v*pc2[i],0)]);
}

function resize() { const r=sc.parentElement.getBoundingClientRect(); sc.width=r.width; sc.height=r.height; }
window.addEventListener('resize', resize);

function w2c(wx,wy) {
  const P=70,W=sc.width,H=sc.height,rx=bounds.maxX-bounds.minX||1,ry=bounds.maxY-bounds.minY||1;
  return [P+((wx-bounds.minX)/rx)*(W-2*P), H-P-((wy-bounds.minY)/ry)*(H-2*P)];
}

function drawFrame() {
  ctx.clearRect(0,0,sc.width,sc.height);
  ctx.fillStyle='#07070f'; ctx.fillRect(0,0,sc.width,sc.height);
  ctx.strokeStyle='#0e0e1e'; ctx.lineWidth=1;
  for (let i=0;i<=8;i++) {
    const tx=70+(i/8)*(sc.width-140),ty=70+(i/8)*(sc.height-140);
    ctx.beginPath();ctx.moveTo(tx,70);ctx.lineTo(tx,sc.height-70);ctx.stroke();
    ctx.beginPath();ctx.moveTo(70,ty);ctx.lineTo(sc.width-70,ty);ctx.stroke();
  }
  ctx.fillStyle='#1a1a38'; ctx.font='11px Fira Code,monospace';
  ctx.fillText('PC₁ →',sc.width/2-40,sc.height-18);
  ctx.save();ctx.translate(18,sc.height/2+50);ctx.rotate(-Math.PI/2);ctx.fillText('PC₂ →',0,0);ctx.restore();
  ctx.fillStyle='#151530'; ctx.font='12px Fira Code,monospace';
  ctx.fillText('2D PCA Projection  ·  Semantic Space',80,28);

  if (queryPt && hitIds.size>0) {
    const [qx,qy]=w2c(queryPt.x,queryPt.y);
    for (const pt of pcaPoints) {
      if (!hitIds.has(pt.item.id)) continue;
      const [px,py]=w2c(pt.x,pt.y);
      ctx.strokeStyle='rgba(108,99,255,0.18)'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
      ctx.beginPath();ctx.moveTo(qx,qy);ctx.lineTo(px,py);ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  for (const pt of pcaPoints) {
    const [cx,cy]=w2c(pt.x,pt.y);
    const col=COL[pt.item.category]||COL.default;
    const isHit=hitIds.has(pt.item.id), r=isHit?10:7;
    if (isHit) {
      const pr=r+7+Math.sin(pulse)*3.5;
      ctx.beginPath();ctx.arc(cx,cy,pr,0,2*Math.PI);
      ctx.strokeStyle=col+'55';ctx.lineWidth=1.5;ctx.stroke();
    }
    const grd=ctx.createRadialGradient(cx,cy,0,cx,cy,r*3);
    grd.addColorStop(0,col+(isHit?'bb':'88'));grd.addColorStop(1,'transparent');
    ctx.beginPath();ctx.arc(cx,cy,r*3,0,2*Math.PI);ctx.fillStyle=grd;ctx.fill();
    ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);ctx.fillStyle=col;ctx.fill();
    if (hoverItem&&hoverItem.id===pt.item.id) {
      ctx.beginPath();ctx.arc(cx,cy,r+5,0,2*Math.PI);ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.stroke();
    }
  }
  if (queryPt) {
    const [qx,qy]=w2c(queryPt.x,queryPt.y);
    ctx.save();ctx.translate(qx,qy);
    ctx.shadowColor='#fff';ctx.shadowBlur=18;
    ctx.beginPath();
    for (let i=0;i<10;i++){const a=(i*Math.PI/5)-Math.PI/2,rr=i%2===0?13:5;if(i===0)ctx.moveTo(Math.cos(a)*rr,Math.sin(a)*rr);else ctx.lineTo(Math.cos(a)*rr,Math.sin(a)*rr);}
    ctx.closePath();ctx.fillStyle='#fff';ctx.fill();
    ctx.shadowBlur=0;ctx.restore();
    ctx.fillStyle='#aaaacc';ctx.font='10px Fira Code,monospace';ctx.fillText('query',qx+16,qy+4);
  }
  if (!pcaPoints.length) {
    ctx.fillStyle='#1a1a38';ctx.font='13px Fira Code,monospace';ctx.textAlign='center';
    ctx.fillText('Connecting to VectorDB…',sc.width/2,sc.height/2);ctx.textAlign='left';
  }
  pulse+=0.05;
  requestAnimationFrame(drawFrame);
}

sc.addEventListener('mousemove', e => {
  const rect=sc.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
  hoverItem=null; let best=18;
  for (const pt of pcaPoints) {
    const [cx,cy]=w2c(pt.x,pt.y),d=Math.hypot(mx-cx,my-cy);
    if (d<best){best=d;hoverItem=pt.item;}
  }
  const tip=document.getElementById('tip');
  if (hoverItem) {
    const col=COL[hoverItem.category]||COL.default;
    tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-8)+'px';
    tip.innerHTML=`<span style="color:${col}">[${hoverItem.category.toUpperCase()}]</span><br>${hoverItem.metadata}`;
  } else tip.style.display='none';
});
sc.addEventListener('mouseleave',()=>{hoverItem=null;document.getElementById('tip').style.display='none';});

async function loadItems() {
  try {
    const r = await fetch(API+'/items');
    allItems = await r.json();
    if (allItems.length >= 2) {
      const coords = pca2D(allItems.map(v=>v.embedding));
      pcaPoints = allItems.map((item,i)=>({x:coords[i][0],y:coords[i][1],item}));
      let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
      for (const p of pcaPoints){x0=Math.min(x0,p.x);x1=Math.max(x1,p.x);y0=Math.min(y0,p.y);y1=Math.max(y1,p.y);}
      const px=(x1-x0)*.18||.1,py=(y1-y0)*.18||.1;
      bounds={minX:x0-px,maxX:x1+px,minY:y0-py,maxY:y1+py};
    }
  } catch(_) {}
}
"""

content = content.replace("// CENTER TAB SWITCH", pca_funcs + "\n// CENTER TAB SWITCH")


# 5. Modify runSearch
runsearch_old = """    const r = await fetch(`${API}/search?v=${emb.join(',')}&k=${k}&metric=${metric}&algo=${selAlgo}`);
    const data = await r.json();
    searchResults = data.results||[];"""

runsearch_new = """    const r = await fetch(`${API}/search?v=${emb.join(',')}&k=${k}&metric=${metric}&algo=${selAlgo}`);
    const data = await r.json();
    searchResults = data.results||[];
    hitIds = new Set(searchResults.map(r=>r.id));
    if (searchResults.length>0){
      let sx=0,sy=0,sw=0;
      for (let i=0;i<Math.min(3,searchResults.length);i++){
        const pt=pcaPoints.find(p=>p.item.id===searchResults[i].id);
        if(pt){const w=1/(i+1);sx+=pt.x*w;sy+=pt.y*w;sw+=w;}
      }
      if(sw>0)queryPt={x:sx/sw+(Math.random()-.5)*.015,y:sy/sw+(Math.random()-.5)*.015};
    }"""
content = content.replace(runsearch_old, runsearch_new)

# Modify askAI
askAI_old = """  try {
    const r = await fetch(API+'/doc/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,k})});"""

askAI_new = """  // Update the scatter plot visualizer in the background
  fetch(API+'/doc/search', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, k})
  })
  .then(res=>res.json())
  .then(data => {
    if (data.contexts && data.contexts.length > 0) {
      hitIds = new Set();
      let sx=0, sy=0, sw=0;
      data.contexts.forEach((ctx, i) => {
        // Match the RAG chunk title back to the visualizer document metadata
        const pt = pcaPoints.find(p => p.item.category === 'doc' && ctx.title.startsWith(p.item.metadata));
        if (pt) {
          hitIds.add(pt.item.id);
          const w = 1/(i+1); sx += pt.x*w; sy += pt.y*w; sw += w;
        }
      });
      if (sw > 0) queryPt = {x: sx/sw + (Math.random()-.5)*.015, y: sy/sw + (Math.random()-.5)*.015};
    } else {
      hitIds = new Set();
      const emb16 = textToEmbedding(question);
      fetch(`${API}/search?v=${emb16.join(',')}&k=3&metric=cosine&algo=hnsw`)
        .then(res2=>res2.json())
        .then(data2 => {
          if (data2.results && data2.results.length>0) {
            let sx=0,sy=0,sw=0;
            for (let i=0;i<Math.min(3,data2.results.length);i++) {
              const pt=pcaPoints.find(p=>p.item.id===data2.results[i].id);
              if(pt){const w=1/(i+1);sx+=pt.x*w;sy+=pt.y*w;sw+=w;}
            }
            if(sw>0) queryPt={x:sx/sw+(Math.random()-.5)*.015,y:sy/sw+(Math.random()-.5)*.015};
          }
        }).catch(()=>{});
    }
  }).catch(()=>{});

  try {
    const r = await fetch(API+'/doc/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,k})});"""

content = content.replace(askAI_old, askAI_new)


# Update addVector
content = content.replace("document.getElementById('addMeta').value='';\n    loadHNSW(); loadStats();", "document.getElementById('addMeta').value='';\n    loadItems().then(loadHNSW); loadStats();")

# Update insertDocument
content = content.replace("      document.getElementById('docText').value='';\n      loadDocList(); loadStats(); checkOllamaStatus();", "      document.getElementById('docText').value='';\n      const emb16 = textToEmbedding(title + ' ' + text);\n      fetch(API+'/insert', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({metadata: title, category: 'doc', embedding: emb16}) }).then(() => { loadItems().then(loadHNSW); });\n      loadDocList(); loadStats(); checkOllamaStatus();")

# Update uploadFile
content = content.replace("      setTimeout(() => statusEl.classList.remove('show'), 3000);\n      loadDocList(); loadStats(); loadHNSW();", "      setTimeout(() => statusEl.classList.remove('show'), 3000);\n      loadDocList(); loadStats(); loadItems().then(loadHNSW);")


# Modify BOOT section
boot_old = """// BOOT
checkOllamaStatus();
loadDocList();
loadStats();
loadHNSW();"""

boot_new = """// BOOT
resize();
drawFrame();
loadItems().then(loadHNSW);
checkOllamaStatus();
loadDocList();
loadStats();"""

content = content.replace(boot_old, boot_new)


with open('src/main/resources/index.html', 'w') as f:
    f.write(content)
