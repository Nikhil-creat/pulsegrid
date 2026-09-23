(function () {

  "use strict";

  // Core algorithm (random sampling + the hand-written MLP) lives in mlp.js
  // so it can be unit-tested in isolation with Node's test runner — see /tests.
  const { gauss, clamp, randInit, makeMLP, tanh, dtanh, sigmoid, forward, trainStep, synthesizeDataset } = window.PulseGridMLP;

  /* =========================================================
     0. THEME + BOOT SEQUENCE
  ========================================================= */
  const root = document.documentElement;
  const savedTheme = localStorage.getItem("pulsegrid-theme");
  if (savedTheme) root.setAttribute("data-theme", savedTheme);
  document.getElementById("themeToggle").addEventListener("click", ()=>{
    const cur = root.getAttribute("data-theme") === "light" ? "light" : "dark";
    const next = cur === "light" ? "dark" : "light";
    if (next === "dark") root.removeAttribute("data-theme"); else root.setAttribute("data-theme","light");
    localStorage.setItem("pulsegrid-theme", next);
  });

  const bootLines = [
    "booting pulsegrid runtime…",
    "calibrating 10 device baselines…",
    "seeding attack simulator…",
    "compiling neural detector (4-8-1)…",
    "linking three.js render pipeline…",
    "checking local storage for saved weights…",
    "mesh online."
  ];
  const bootLogEl = document.getElementById("bootLog");
  const bootBar = document.getElementById("bootBar");
  const bootCredit = document.getElementById("bootCredit");
  const bootEl = document.getElementById("boot");
  let bootDone = false;

  function finishBoot(){
    if (bootDone) return;
    bootDone = true;
    bootEl.classList.add("hidden");
  }
  document.getElementById("bootSkip").addEventListener("click", finishBoot);

  bootLines.forEach((line, i)=>{
    setTimeout(()=>{
      const div = document.createElement("div");
      div.className = "c-line";
      div.innerHTML = '<span class="ok">✓</span> ' + line;
      bootLogEl.appendChild(div);
      bootBar.style.width = (((i+1)/bootLines.length)*100).toFixed(0) + "%";
      if (i === bootLines.length-1) bootCredit.classList.add("show");
    }, 260*i);
  });
  setTimeout(finishBoot, 260*bootLines.length + 1400);

  /* =========================================================
     0b. SYSTEM CONSOLE
  ========================================================= */
  const consoleEl = document.getElementById("console");
  function logConsole(msg, level){
    const line = document.createElement("div");
    line.className = "c-line";
    const time = new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
    const cls = level === "crit" ? "c-crit" : level === "warn" ? "c-warn" : level === "ok" ? "c-ok" : "";
    line.innerHTML = '<span class="c-time">'+time+'</span><span class="'+cls+'">'+msg+'</span>';
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
    while (consoleEl.children.length > 200) consoleEl.removeChild(consoleEl.firstChild);
  }
  logConsole("pulsegrid runtime initialised", "ok");

  /* =========================================================
     0c. SOUND ALERTS
  ========================================================= */
  let soundOn = false;
  let audioCtx = null;
  const soundToggle = document.getElementById("soundToggle");
  soundToggle.addEventListener("click", ()=>{
    soundOn = !soundOn;
    soundToggle.style.color = soundOn ? "var(--brand)" : "";
    soundToggle.style.borderColor = soundOn ? "var(--brand)" : "";
    logConsole(soundOn ? "sound alerts enabled" : "sound alerts disabled");
  });
  function beep(freq){
    if (!soundOn) return;
    try{
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine"; osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.06, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.35);
      osc.connect(gain); gain.connect(audioCtx.destination);
      osc.start(); osc.stop(audioCtx.currentTime + 0.35);
    }catch(e){ /* audio unavailable, ignore */ }
  }

  /* =========================================================
     1. DEVICE MESH SIMULATION
  ========================================================= */
  const DEVICE_TYPES = ["Temp sensor","Motion sensor","Door lock","Camera node","Smart plug","Humidity sensor","Access panel","Relay switch","Air monitor","Edge camera"];
  const N_DEVICES = 10;
  let attackProbability = 0.08;

  function makeDevice(i){
    return {
      id: "sensor-" + String(i+1).padStart(2,"0"),
      type: DEVICE_TYPES[i % DEVICE_TYPES.length],
      baseline: {
        traffic: gauss(40, 8),
        latency: gauss(60, 12),
        authFail: gauss(0.02, 0.01),
        jitter: gauss(3, 1)
      },
      state: { traffic:0, latency:0, authFail:0, jitter:0 },
      attackTicksLeft: 0,
      score: 0.05 + Math.random()*0.05,
      scoreHistory: [],
      angle: (i / N_DEVICES) * Math.PI * 2,
      lastFlaggedAt: 0
    };
  }

  const devices = Array.from({length:N_DEVICES}, (_,i)=>makeDevice(i));

  const attackSelect = document.getElementById("attackSelect");
  devices.forEach(d=>{
    const opt = document.createElement("option");
    opt.value = d.id; opt.textContent = d.id + " — " + d.type;
    attackSelect.appendChild(opt);
  });
  document.getElementById("attackBtn").addEventListener("click", ()=>{
    const d = devices.find(x=>x.id === attackSelect.value);
    if (d){
      d.attackTicksLeft = 4 + Math.floor(Math.random()*4);
      logConsole("manual attack triggered on " + d.id, "warn");
    }
  });

  const speedRange = document.getElementById("speedRange");
  const speedValue = document.getElementById("speedValue");
  speedRange.addEventListener("input", ()=>{
    speedValue.textContent = (speedRange.value/1000).toFixed(1) + "s / tick";
    restartTickLoop();
  });

  const attackRateRange = document.getElementById("attackRateRange");
  attackRateRange.addEventListener("input", ()=>{
    attackProbability = attackRateRange.value/100;
    document.getElementById("attackRateValue").textContent = attackRateRange.value + "%";
  });

  function sampleDevice(d){
    const attacking = d.attackTicksLeft > 0;
    if (attacking) d.attackTicksLeft--;
    const b = d.baseline;
    const mult = attacking ? gauss(3.2, 0.6) : 1;
    return {
      traffic: Math.max(0, gauss(b.traffic, b.traffic*0.15) * mult),
      latency: Math.max(1, gauss(b.latency, b.latency*0.12) * (attacking? gauss(2.1,0.3):1)),
      authFail: clamp(gauss(b.authFail, 0.01) + (attacking? gauss(0.5,0.15):0), 0, 1),
      jitter: Math.max(0, gauss(b.jitter, b.jitter*0.2) * (attacking? gauss(3.5,0.5):1)),
      attacking
    };
  }

  function maybeTriggerAttack(){
    if (Math.random() < attackProbability){
      const d = devices[Math.floor(Math.random()*devices.length)];
      if (d.attackTicksLeft <= 0){
        d.attackTicksLeft = 3 + Math.floor(Math.random()*4);
        logConsole(d.id + " entering anomalous state", "warn");
      }
    }
  }

  /* =========================================================
     2. FEATURE NORMALISATION + HEURISTIC FALLBACK
  ========================================================= */
  function toFeatureVector(d, s){
    const b = d.baseline;
    return [
      (s.traffic - b.traffic) / (b.traffic*0.4 + 1e-6),
      (s.latency - b.latency) / (b.latency*0.4 + 1e-6),
      (s.authFail - b.authFail) / 0.25,
      (s.jitter - b.jitter) / (b.jitter*0.6 + 1e-6)
    ];
  }
  function heuristicScore(vec){
    const mag = Math.sqrt(vec.reduce((a,v)=>a+v*v,0)) / 2;
    return clamp(mag, 0, 1);
  }

  /* =========================================================
     3. DETECTOR STATE
     (forward / trainStep / synthesizeDataset now come from
     window.PulseGridMLP — see assets/js/mlp.js and /tests/mlp.test.js)
  ========================================================= */
  let net = makeMLP(4, 8, 1);
  let netTrained = false;

  /* =========================================================
     4. TRAINING UI + SAVE/LOAD WEIGHTS
  ========================================================= */
  const trainBtn = document.getElementById("trainBtn");
  const saveBtn = document.getElementById("saveBtn");
  const mEpoch = document.getElementById("mEpoch");
  const mLoss = document.getElementById("mLoss");
  const mAcc = document.getElementById("mAcc");
  const meshBadge = document.getElementById("meshBadge");
  const lossCanvas = document.getElementById("lossCanvas");
  const lctx = lossCanvas.getContext("2d");
  let lossHistory = [];

  function activateBadge(){
    meshBadge.className = "badge active";
    meshBadge.innerHTML = '<span class="dot"></span> neural net active';
  }

  function drawLossCurve(){
    const w = lossCanvas.width, h = lossCanvas.height;
    lctx.clearRect(0,0,w,h);
    lctx.strokeStyle = "#1C2733"; lctx.lineWidth = 1;
    for (let i=1;i<4;i++){
      const y = (h/4)*i;
      lctx.beginPath(); lctx.moveTo(0,y); lctx.lineTo(w,y); lctx.stroke();
    }
    if (lossHistory.length < 2) return;
    const maxLoss = Math.max(...lossHistory, 0.05);
    lctx.strokeStyle = "#7C5CFC";
    lctx.lineWidth = 1.6;
    lctx.beginPath();
    lossHistory.forEach((v,i)=>{
      const x = (i/(lossHistory.length-1)) * w;
      const y = h - (v/maxLoss) * (h-8) - 4;
      if (i===0) lctx.moveTo(x,y); else lctx.lineTo(x,y);
    });
    lctx.stroke();
  }
  drawLossCurve();

  const EPOCHS = 220;
  function trainModelAsync(){
    trainBtn.disabled = true;
    trainBtn.textContent = "Training…";
    logConsole("training started — 500 synthetic samples, 220 epochs", "warn");
    lossHistory = [];
    const { X, Y } = synthesizeDataset(500);
    const splitAt = 400;
    const trainX = X.slice(0,splitAt), trainY = Y.slice(0,splitAt);
    const testX = X.slice(splitAt), testY = Y.slice(splitAt);
    net = makeMLP(4,8,1);
    let epoch = 0;

    function chunk(){
      const stepsThisFrame = 2;
      for (let s=0; s<stepsThisFrame && epoch<EPOCHS; s++, epoch++){
        let epochLoss = 0;
        for (let i=0;i<trainX.length;i++) epochLoss += trainStep(net, trainX[i], trainY[i], 0.06);
        epochLoss /= trainX.length;
        lossHistory.push(epochLoss);
        mEpoch.textContent = epoch+1 + " / " + EPOCHS;
        mLoss.textContent = epochLoss.toFixed(4);
      }
      drawLossCurve();
      if (epoch < EPOCHS){
        requestAnimationFrame(chunk);
      } else {
        let correct = 0;
        testX.forEach((x,i)=>{
          const { o } = forward(net, x);
          if ((o[0] > 0.5 ? 1 : 0) === testY[i][0]) correct++;
        });
        mAcc.textContent = ((correct/testX.length)*100).toFixed(1) + "%";
        netTrained = true;
        trainBtn.disabled = false;
        trainBtn.textContent = "Retrain on a fresh 500 samples";
        activateBadge();
        logConsole("training complete — holdout accuracy " + mAcc.textContent, "ok");
      }
    }
    requestAnimationFrame(chunk);
  }
  trainBtn.addEventListener("click", trainModelAsync);

  saveBtn.addEventListener("click", ()=>{
    if (!netTrained){ logConsole("nothing to save yet — train the net first", "warn"); return; }
    try{
      localStorage.setItem("pulsegrid-weights", JSON.stringify(net));
      logConsole("trained weights saved to this browser", "ok");
    }catch(e){ logConsole("could not save weights (storage unavailable)", "warn"); }
  });

  (function tryLoadSavedWeights(){
    try{
      const raw = localStorage.getItem("pulsegrid-weights");
      if (raw){
        const parsed = JSON.parse(raw);
        if (parsed && parsed.W1 && parsed.W2){
          net = parsed;
          netTrained = true;
          mAcc.textContent = "loaded from browser";
          activateBadge();
          logConsole("restored previously trained weights from local storage", "ok");
        }
      }
    }catch(e){ /* ignore malformed storage */ }
  })();

  /* =========================================================
     5. THREE.JS MESH VISUALISATION
  ========================================================= */
  const holder = document.getElementById("meshCanvasHolder");
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(48, holder.clientWidth/holder.clientHeight, 0.1, 100);
  camera.position.set(0, 6.5, 13);
  camera.lookAt(0,0,0);

  const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
  renderer.setSize(holder.clientWidth, holder.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  holder.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0x8891a8, 0.9));
  const pLight = new THREE.PointLight(0x7c5cfc, 1.4, 30);
  pLight.position.set(4,8,6);
  scene.add(pLight);
  const pLight2 = new THREE.PointLight(0x3fc1b0, 0.7, 30);
  pLight2.position.set(-6,-2,-4);
  scene.add(pLight2);

  const group = new THREE.Group();
  scene.add(group);

  const hubGeo = new THREE.IcosahedronGeometry(0.85, 1);
  const hubMat = new THREE.MeshStandardMaterial({ color:0x7c5cfc, emissive:0x2a1f6b, roughness:0.35, metalness:0.3 });
  const hub = new THREE.Mesh(hubGeo, hubMat);
  group.add(hub);

  const RADIUS = 5.4;
  const nodeGeo = new THREE.SphereGeometry(0.34, 24, 24);
  const nodeMeshes = [];
  const edgeLines = [];

  function colorForScore(score){
    if (score < 0.33) return new THREE.Color(0x3fc1b0);
    if (score < 0.66) return new THREE.Color(0xe8a33d);
    return new THREE.Color(0xe1594f);
  }

  devices.forEach((d)=>{
    const x = Math.cos(d.angle) * RADIUS;
    const z = Math.sin(d.angle) * RADIUS;
    const mat = new THREE.MeshStandardMaterial({ color:0x3fc1b0, roughness:0.4, metalness:0.2 });
    const mesh = new THREE.Mesh(nodeGeo, mat);
    mesh.position.set(x, 0, z);
    mesh.userData.device = d;
    group.add(mesh);
    nodeMeshes.push(mesh);

    const points = [new THREE.Vector3(0,0,0), new THREE.Vector3(x,0,z)];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const lineMat = new THREE.LineBasicMaterial({ color:0x2a3644, transparent:true, opacity:0.7 });
    const line = new THREE.Line(geo, lineMat);
    group.add(line);
    edgeLines.push(line);
  });

  function onResize(){
    const w = holder.clientWidth, h = holder.clientHeight;
    camera.aspect = w/h; camera.updateProjectionMatrix();
    renderer.setSize(w,h);
  }
  window.addEventListener("resize", onResize);

  // manual drag-to-orbit (lightweight, no OrbitControls dependency)
  let dragging = false, lastX = 0, lastY = 0, manualYaw = 0, manualPitch = 0.45, autoSpin = true;
  renderer.domElement.addEventListener("mousedown", (e)=>{ dragging = true; autoSpin = false; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", ()=> dragging = false);
  window.addEventListener("mousemove", (e)=>{
    if (!dragging) return;
    manualYaw += (e.clientX - lastX) * 0.006;
    manualPitch = clamp(manualPitch + (e.clientY - lastY) * 0.004, 0.05, 1.3);
    lastX = e.clientX; lastY = e.clientY;
  });

  // hover + click
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const hoverCard = document.getElementById("hoverCard");
  const hcTitle = document.getElementById("hcTitle");
  const hcScore = document.getElementById("hcScore");
  const hcTraffic = document.getElementById("hcTraffic");
  const hcAuth = document.getElementById("hcAuth");

  renderer.domElement.addEventListener("mousemove", (e)=>{
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX-rect.left)/rect.width)*2-1;
    mouse.y = -((e.clientY-rect.top)/rect.height)*2+1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(nodeMeshes);
    if (hits.length && !dragging){
      const d = hits[0].object.userData.device;
      hcTitle.textContent = d.id + " · " + d.type;
      hcScore.textContent = (d.score*100).toFixed(0) + "%";
      hcTraffic.textContent = d.state.traffic.toFixed(1) + " req/min";
      hcAuth.textContent = (d.state.authFail*100).toFixed(1) + "%";
      hoverCard.style.opacity = 1;
      hoverCard.style.left = (e.clientX-rect.left+16)+"px";
      hoverCard.style.top = (e.clientY-rect.top+16)+"px";
    } else {
      hoverCard.style.opacity = 0;
    }
  });
  renderer.domElement.addEventListener("mouseleave", ()=> hoverCard.style.opacity = 0);
  renderer.domElement.addEventListener("click", (e)=>{
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX-rect.left)/rect.width)*2-1;
    mouse.y = -((e.clientY-rect.top)/rect.height)*2+1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(nodeMeshes);
    if (hits.length) openModal(hits[0].object.userData.device);
  });

  let t = 0;
  function animate(){
    requestAnimationFrame(animate);
    t += 0.006;
    if (autoSpin) manualYaw = t;
    group.rotation.y = manualYaw;
    camera.position.set(Math.sin(manualPitch)*13*0 + 0, 6.5 + (manualPitch-0.45)*6, 13);
    camera.lookAt(0,0,0);
    hub.rotation.y -= 0.01;
    hub.rotation.x += 0.004;
    nodeMeshes.forEach((mesh)=>{
      const d = mesh.userData.device;
      const target = colorForScore(d.score);
      mesh.material.color.lerp(target, 0.08);
      const pulse = d.score > 0.66 ? 1 + Math.sin(t*10)*0.12 : 1;
      mesh.scale.setScalar(pulse);
    });
    edgeLines.forEach((line, i)=>{
      const d = devices[i];
      const c = colorForScore(d.score);
      line.material.color.lerp(c, 0.08);
      line.material.opacity = 0.35 + d.score*0.55;
    });
    renderer.render(scene, camera);
  }
  animate();

  /* =========================================================
     5b. DEVICE MODAL + SPARKLINE
  ========================================================= */
  const modalOverlay = document.getElementById("modalOverlay");
  const sparkCanvas = document.getElementById("sparkCanvas");
  const sctx = sparkCanvas.getContext("2d");
  let modalDevice = null;

  function drawSpark(d){
    const w = sparkCanvas.width, h = sparkCanvas.height;
    sctx.clearRect(0,0,w,h);
    const hist = d.scoreHistory;
    if (hist.length < 2) return;
    sctx.strokeStyle = "#7C5CFC"; sctx.lineWidth = 2;
    sctx.beginPath();
    hist.forEach((v,i)=>{
      const x = (i/(hist.length-1))*w;
      const y = h - v*(h-6) - 3;
      if (i===0) sctx.moveTo(x,y); else sctx.lineTo(x,y);
    });
    sctx.stroke();
  }

  function refreshModal(){
    if (!modalDevice) return;
    const d = modalDevice;
    document.getElementById("modalTitle").textContent = d.id;
    document.getElementById("modalType").textContent = d.type;
    document.getElementById("modalScore").textContent = (d.score*100).toFixed(0) + "%";
    const sevEl = document.getElementById("modalSev");
    const cls = d.score<0.33?"safe":d.score<0.66?"warn":"crit";
    sevEl.className = "sev " + cls;
    sevEl.textContent = d.score<0.33?"normal":d.score<0.66?"suspicious":"critical";
    document.getElementById("modalTraffic").textContent = d.state.traffic.toFixed(1) + " req/min";
    document.getElementById("modalLatency").textContent = d.state.latency.toFixed(1) + " ms";
    document.getElementById("modalAuth").textContent = (d.state.authFail*100).toFixed(1) + "%";
    document.getElementById("modalJitter").textContent = d.state.jitter.toFixed(2) + " ms";
    drawSpark(d);
  }
  function openModal(d){ modalDevice = d; refreshModal(); modalOverlay.classList.add("show"); }
  function closeModal(){ modalDevice = null; modalOverlay.classList.remove("show"); }
  document.getElementById("modalClose").addEventListener("click", closeModal);
  modalOverlay.addEventListener("click", (e)=>{ if (e.target === modalOverlay) closeModal(); });

  /* =========================================================
     6. TELEMETRY CHART
  ========================================================= */
  const ctx = document.getElementById("telemetryChart").getContext("2d");
  const telemetryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "avg traffic (req/min)", data: [], borderColor: "#3fc1b0",
        backgroundColor: "#3fc1b022", fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2
      },{
        label: "avg threat score ×100", data: [], borderColor: "#e1594f",
        backgroundColor: "#e1594f11", fill: false, tension: 0.35, pointRadius: 0, borderWidth: 2
      }]
    },
    options: {
      responsive: true, animation: false, interaction: { mode:"index", intersect:false },
      scales: {
        x: { ticks:{ color:"#5B6879", maxTicksLimit:6, font:{family:"IBM Plex Mono", size:10} }, grid:{ color:"#1C2733" } },
        y: { ticks:{ color:"#5B6879", font:{family:"IBM Plex Mono", size:10} }, grid:{ color:"#1C2733" } }
      },
      plugins: { legend: { labels:{ color:"#8B99AB", font:{family:"IBM Plex Mono", size:10.5}, boxWidth:10 } } }
    }
  });

  /* =========================================================
     7. THREAT LOG + CSV EXPORT
  ========================================================= */
  const logBody = document.getElementById("logBody");
  const logEmpty = document.getElementById("logEmpty");
  const MAX_LOG_ROWS = 40;
  const logEntries = [];

  function sevClass(score){ return score<0.33?"safe":score<0.66?"warn":"crit"; }
  function sevLabel(score){ return score<0.33?"normal":score<0.66?"suspicious":"critical"; }
  function likelyCause(vec){
    const names = ["request rate","latency","auth failures","jitter"];
    let maxI = 0;
    for (let i=1;i<vec.length;i++) if (Math.abs(vec[i])>Math.abs(vec[maxI])) maxI = i;
    return names[maxI] + " spike";
  }

  function logRow(d, score, vec){
    logEmpty.style.display = "none";
    const tr = document.createElement("tr");
    const now = new Date();
    const time = now.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit", second:"2-digit"});
    const cause = likelyCause(vec);
    tr.innerHTML =
      "<td>"+time+"</td><td>"+d.id+"</td><td>"+d.type+"</td>" +
      "<td><span class='sev "+sevClass(score)+"'>"+sevLabel(score)+"</span></td>" +
      "<td>"+(score*100).toFixed(0)+"%</td><td>"+cause+"</td>";
    tr.addEventListener("click", ()=> openModal(d));
    logBody.prepend(tr);
    while (logBody.children.length > MAX_LOG_ROWS) logBody.removeChild(logBody.lastChild);
    logEntries.unshift({ time, id:d.id, type:d.type, severity:sevLabel(score), score:(score*100).toFixed(1), cause });
    if (score > 0.66) beep(880); else beep(520);
  }

  document.getElementById("exportBtn").addEventListener("click", ()=>{
    if (!logEntries.length){ logConsole("nothing to export yet", "warn"); return; }
    const header = "time,device,type,severity,score_pct,likely_cause\n";
    const rows = logEntries.map(r=> [r.time,r.id,r.type,r.severity,r.score,r.cause].join(",")).join("\n");
    const blob = new Blob([header+rows], { type:"text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "pulsegrid-threat-log.csv";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
    logConsole("exported " + logEntries.length + " log rows to CSV", "ok");
  });

  /* =========================================================
     8. MAIN TICK LOOP
  ========================================================= */
  const gaugeArc = document.getElementById("gaugeArc");
  const gaugeValue = document.getElementById("gaugeValue");
  const gaugeTrend = document.getElementById("gaugeTrend");
  const CIRC = 150.8;
  let scoreTrendBuffer = [];

  function tick(){
    maybeTriggerAttack();
    let sumTraffic = 0, sumScore = 0;

    devices.forEach((d)=>{
      const s = sampleDevice(d);
      d.state = s;
      const vec = toFeatureVector(d, s);
      let score;
      if (netTrained){
        const { o } = forward(net, vec);
        score = clamp(o[0], 0, 1);
      } else {
        score = heuristicScore(vec);
      }
      d.score = d.score*0.5 + score*0.5;
      d.scoreHistory.push(d.score);
      if (d.scoreHistory.length > 30) d.scoreHistory.shift();
      sumTraffic += s.traffic;
      sumScore += d.score;

      if (d.score > 0.33 && Date.now() - d.lastFlaggedAt > 1500){
        logRow(d, d.score, vec);
        d.lastFlaggedAt = Date.now();
      }
    });

    if (modalDevice) refreshModal();

    const avgTraffic = sumTraffic / devices.length;
    const avgScore = sumScore / devices.length;

    scoreTrendBuffer.push(avgScore);
    if (scoreTrendBuffer.length > 6) scoreTrendBuffer.shift();
    if (scoreTrendBuffer.length >= 4){
      const delta = scoreTrendBuffer[scoreTrendBuffer.length-1] - scoreTrendBuffer[0];
      gaugeTrend.textContent = delta > 0.04 ? "rising" : delta < -0.04 ? "falling" : "steady";
    }

    const pct = clamp(avgScore, 0, 1);
    gaugeArc.setAttribute("stroke-dashoffset", (CIRC * (1-pct)).toFixed(1));
    gaugeArc.setAttribute("stroke", pct<0.33 ? "#3fc1b0" : pct<0.66 ? "#e8a33d" : "#e1594f");
    gaugeValue.textContent = Math.round(pct*100) + "%";
    gaugeValue.className = "g-value " + (pct<0.33?"safe":pct<0.66?"warn":"crit");

    const label = new Date().toLocaleTimeString([], {minute:"2-digit", second:"2-digit"});
    telemetryChart.data.labels.push(label);
    telemetryChart.data.datasets[0].data.push(avgTraffic.toFixed(1));
    telemetryChart.data.datasets[1].data.push((avgScore*100).toFixed(1));
    if (telemetryChart.data.labels.length > 40){
      telemetryChart.data.labels.shift();
      telemetryChart.data.datasets.forEach(ds=>ds.data.shift());
    }
    telemetryChart.update("none");
  }

  let tickTimer = null;
  function restartTickLoop(){
    if (tickTimer) clearInterval(tickTimer);
    tickTimer = setInterval(tick, Number(speedRange.value));
  }
  restartTickLoop();
  tick();
}());
