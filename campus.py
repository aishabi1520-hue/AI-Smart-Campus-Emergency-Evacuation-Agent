"""
AI-Based Smart Campus Emergency Response and Evacuation Agent
-------------------------------------------------------------
Run:   python campus_evacuation_agent.py
Opens: http://localhost:8000   (no pip install needed - only Python 3.8+)

The agent: Perceive (incident input) -> Assess (risk score) -> Plan (Dijkstra
shortest safe path + assembly-point capacity) -> Act (alerts, services, animation).
"""
import heapq, json, math, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8000
M_PER_UNIT = 0.4  # 1 map unit = 0.4 metres

# ---------------------------------------------------------------- campus graph
NODES = {
    "MAIN": dict(name="Main Block", x=170, y=110, t="b", pop=320),
    "LIB": dict(name="Library", x=440, y=90, t="b", pop=180),
    "LAB": dict(name="CS Labs", x=730, y=110, t="b", pop=240),
    "AUD": dict(name="Auditorium", x=120, y=330, t="b", pop=400),
    "CAF": dict(name="Cafeteria", x=440, y=320, t="b", pop=150),
    "HOS": dict(name="Hostel", x=740, y=330, t="b", pop=350),
    "J1": dict(name="Junction J1", x=300, y=200, t="j"),
    "J2": dict(name="Junction J2", x=590, y=200, t="j"),
    "J3": dict(name="Junction J3", x=290, y=330, t="j"),
    "J4": dict(name="Junction J4", x=590, y=330, t="j"),
    "ASM_A": dict(name="Assembly A (Front Lawn)", x=290, y=470, t="a", cap=900),
    "ASM_B": dict(name="Assembly B (Sports Ground)", x=640, y=470, t="a", cap=1000),
}
EDGES = [("MAIN", "J1"), ("LIB", "J1"), ("LIB", "J2"), ("LAB", "J2"), ("J1", "J2"),
         ("J1", "J3"), ("J2", "J4"), ("J3", "CAF"), ("CAF", "J4"), ("AUD", "J3"),
         ("HOS", "J4"), ("J3", "ASM_A"), ("J4", "ASM_B"), ("CAF", "ASM_A"),
         ("HOS", "ASM_B"), ("AUD", "ASM_A")]

ADJ = {k: [] for k in NODES}
for a, b in EDGES:
    w = math.hypot(NODES[a]["x"] - NODES[b]["x"], NODES[a]["y"] - NODES[b]["y"])
    ADJ[a].append((b, w))
    ADJ[b].append((a, w))

TYPES = {
    "fire": ("Fire", ["Fire & Rescue (101)", "Ambulance (108)", "Campus Security"],
             "Use stairs, not lifts. Stay low under smoke."),
    "quake": ("Earthquake", ["Disaster Response Force (112)", "Ambulance (108)", "Structural Engineers"],
              "Drop, cover, hold. Move to open ground after shaking stops."),
    "gas": ("Gas leak", ["Hazmat Team", "Fire & Rescue (101)", "Ambulance (108)"],
            "Do not switch on electrical items. Move upwind."),
    "flood": ("Flood", ["Disaster Management (112)", "Water Rescue Team", "Ambulance (108)"],
              "Avoid flooded roads. Move to higher ground."),
}


def dijkstra(src, blocked):
    """Shortest path from src avoiding blocked nodes (buildings are not used as transit)."""
    dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        if u != src and NODES[u]["t"] == "b":
            continue
        for v, w in ADJ[u]:
            if v in blocked:
                continue
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def build_path(prev, src, dst):
    p = [dst]
    while p[-1] != src:
        p.append(prev[p[-1]])
    return p[::-1]


def simulate(d):
    hz = d.get("location", "LAB")
    sev = max(1, min(5, int(d.get("severity", 3))))
    etype = d.get("type", "fire")
    label, services, advice = TYPES.get(etype, TYPES["fire"])
    manual = {x for x in d.get("blocked", []) if x in NODES}
    people = {k: max(0, int(d.get("people", {}).get(k, v["pop"])))
              for k, v in NODES.items() if v["t"] == "b"}

    zone = {hz}
    if sev >= 4:  # severe: the roads next to the incident are unsafe too
        zone |= {n for n, _ in ADJ[hz] if NODES[n]["t"] == "j"}
    blocked_all = manual | zone

    caps = {k: v["cap"] for k, v in NODES.items() if v["t"] == "a"}
    load = {k: 0 for k in caps}
    order = [hz] + sorted([b for b in people if b != hz], key=lambda b: -people[b])
    delay = 10 + 3 * sev
    speed = 1.4 * (1 - 0.07 * sev)  # panic slows people down

    results = []
    for b in order:
        blocked = (manual - {b}) if b == hz else (blocked_all - {b})
        dist, prev = dijkstra(b, blocked)
        reach = sorted([(dist[a], a) for a in caps if a in dist])
        r = dict(id=b, name=NODES[b]["name"], people=people[b], hazard=(b == hz),
                 status="empty" if people[b] == 0 else "trapped", assembly=None,
                 path=[], coords=[], meters=0, walk_s=0, eta_s=0)
        if reach and people[b] > 0:
            pick = next((x for x in reach if caps[x[1]] - load[x[1]] >= people[b]), reach[0])
            a = pick[1]
            load[a] += people[b]
            path = build_path(prev, b, a)
            meters = round(pick[0] * M_PER_UNIT)
            walk = round(meters / speed)
            r.update(status="evacuating", assembly=a, path=path, meters=meters, walk_s=walk,
                     eta_s=walk + delay,
                     coords=[[NODES[n]["x"], NODES[n]["y"]] for n in path])
        results.append(r)

    total = sum(people.values())
    trapped = sum(r["people"] for r in results if r["status"] == "trapped")
    safe = total - trapped
    max_eta = max([r["eta_s"] for r in results] + [0])
    score = min(100, sev * 14 + round(30 * trapped / max(total, 1)) + min(20, total // 100)
                + 4 * len(manual))
    level = "Critical" if score >= 70 else "High" if score >= 45 else "Moderate"

    steps = [
        f"Perceive: {label} reported at {NODES[hz]['name']}, severity {sev}/5.",
        f"Assess: {total} people on campus, risk score {score}/100 ({level}).",
        f"Contain: marked {', '.join(NODES[z]['name'] for z in sorted(blocked_all))} as unsafe."
        if blocked_all else "Contain: no roads blocked.",
        "Plan: ran Dijkstra from every building to find the shortest safe route.",
        "Balance: assigned each group to an assembly point with free capacity.",
        f"Act: sent alerts and called {', '.join(services)}.",
    ]
    narr = (f"Alert. {label} reported at {NODES[hz]['name']}, severity {sev} out of 5. "
            f"Risk level is {level}. The agent has blocked the danger zone and planned safe routes "
            f"for {total} people. {safe} people are being guided to assembly points, and full "
            f"evacuation will take about {max_eta} seconds. ")
    if trapped:
        narr += f"Warning. {trapped} people have no safe route and are marked for rescue. "
    narr += f"{services[0]} has been notified."

    return dict(
        results=results, total=total, safe=safe, trapped=trapped, max_eta=max_eta,
        score=score, level=level, blocked=sorted(blocked_all), hazard=hz, type=etype,
        label=label, steps=steps, services=services, narration=narr,
        loads=[dict(id=k, name=NODES[k]["name"], load=load[k], cap=caps[k]) for k in caps],
        alert=f"CAMPUS ALERT: {label} at {NODES[hz]['name']}. Leave calmly by the marked route "
              f"to your assembly point. {advice}",
    )


# ---------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/map":
            self._send(json.dumps(dict(nodes=NODES, edges=EDGES, types={k: v[0] for k, v in TYPES.items()})))
        else:
            self._send(PAGE, "text/html")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self._send(json.dumps(simulate(json.loads(self.rfile.read(n) or b"{}"))))

    def log_message(self, *a):
        pass


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Campus Emergency Response Agent</title>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700&family=Sora:wght@500;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#060a17;--panel:rgba(18,26,52,.72);--line:rgba(130,160,255,.16);--tx:#e8edff;--mut:#8b97c4;
--cy:#2dd4ee;--vi:#8b7bff;--rd:#ff4d6d;--gr:#34e0a1;--am:#ffc247}
*{box-sizing:border-box;margin:0}
body{font-family:Manrope,system-ui,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;
background-image:radial-gradient(900px 500px at 8% -10%,rgba(139,123,255,.28),transparent),
radial-gradient(800px 500px at 100% 0,rgba(45,212,238,.18),transparent),
radial-gradient(700px 500px at 50% 120%,rgba(255,77,109,.14),transparent)}
h1,h2,h3,.big{font-family:Sora,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:18px 26px;flex-wrap:wrap}
.brand{display:flex;gap:14px;align-items:center}
.logo{width:48px;height:48px;border-radius:14px;display:grid;place-items:center;font-size:24px;
background:linear-gradient(135deg,var(--vi),var(--cy));box-shadow:0 0 28px rgba(45,212,238,.4)}
h1{font-size:21px;font-weight:800;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px;margin-top:3px}
button{font:inherit;font-weight:700;border:0;border-radius:12px;padding:11px 16px;cursor:pointer;color:var(--tx);transition:.18s}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--cy);outline-offset:2px}
.primary{background:linear-gradient(135deg,#ff4d6d,#ff8a4c);box-shadow:0 8px 24px rgba(255,77,109,.35);width:100%;font-size:15px}
.primary:hover{transform:translateY(-1px)}.primary:disabled{opacity:.5;cursor:wait}
.ghost{background:rgba(255,255,255,.07);border:1px solid var(--line)}.ghost:hover{background:rgba(255,255,255,.13)}
.gold{background:linear-gradient(135deg,var(--vi),var(--cy));color:#04101f}
main{display:grid;grid-template-columns:300px minmax(0,1fr) 330px;gap:16px;padding:0 26px 30px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:18px;backdrop-filter:blur(14px)}
.card+.card{margin-top:14px}
h3{font-size:15px;margin-bottom:12px}
label{display:block;font-size:12.5px;color:var(--mut);margin:14px 0 6px}
select,input[type=number]{width:100%;background:rgba(5,9,22,.7);border:1px solid var(--line);color:var(--tx);
border-radius:10px;padding:10px;font:inherit}
input[type=range]{width:100%;accent-color:var(--rd)}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{padding:7px 12px;border-radius:99px;border:1px solid var(--line);font-size:13px;cursor:pointer;user-select:none;background:rgba(255,255,255,.04)}
.chip.on{background:rgba(255,77,109,.2);border-color:var(--rd);color:#ffd5dc}
.occ{display:grid;grid-template-columns:1fr 1fr;gap:8px}.occ span{font-size:11.5px;color:var(--mut)}
.occ input{padding:7px 9px;margin-top:3px}
.btns{display:grid;gap:10px;margin-top:18px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:14px 16px}
.kpi small{color:var(--mut);font-size:12px}.kpi .big{font-size:27px;font-weight:800;margin-top:4px}
.bar{height:6px;border-radius:9px;background:rgba(255,255,255,.08);margin-top:8px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cy),var(--gr));transition:width .3s}
svg#map{width:100%;height:auto;display:block;border-radius:16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin-top:10px}
.legend b{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--mut);font-weight:500;padding:6px 8px}
td{padding:8px;border-top:1px solid var(--line)}
.tag{padding:3px 9px;border-radius:99px;font-size:11.5px;font-weight:700}
.t-ev{background:rgba(52,224,161,.16);color:var(--gr)}.t-tr{background:rgba(255,77,109,.18);color:var(--rd)}.t-em{background:rgba(255,255,255,.08);color:var(--mut)}
ol{padding-left:18px;font-size:13px;line-height:1.55;color:#cfd7f7}li{margin-bottom:6px;animation:in .4s}
@keyframes in{from{opacity:0;transform:translateX(8px)}}
.svc{display:flex;gap:8px;flex-wrap:wrap}.svc span{font-size:12.5px;padding:6px 11px;border-radius:10px;background:rgba(255,194,71,.13);color:var(--am)}
.alert{margin-top:12px;padding:12px;border-radius:12px;font-size:12.5px;line-height:1.5;background:rgba(255,77,109,.12);border:1px solid rgba(255,77,109,.35)}
.al{margin-bottom:10px;font-size:12.5px}.al div{display:flex;justify-content:space-between;margin-bottom:4px}
#log{max-height:170px;overflow:auto;font-size:12.5px;color:#b9c3ec;line-height:1.6}#log b{color:var(--cy);font-weight:500}
.pulse{transform-box:fill-box;transform-origin:center;animation:pl 1.4s ease-out infinite}
@keyframes pl{from{transform:scale(.5);opacity:.9}to{transform:scale(2.4);opacity:0}}
.flow{stroke-dasharray:9 9;animation:fl 1s linear infinite}@keyframes fl{to{stroke-dashoffset:-18}}
.shake{animation:sk .5s infinite}@keyframes sk{50%{opacity:.35}}
@media(max-width:1180px){main{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body>
<header>
 <div class="brand"><div class="logo">🛡️</div><div><h1>Smart Campus Emergency Response Agent</h1>
 <div class="sub">AI-based evacuation planning with live simulation</div></div></div>
 <div style="display:flex;gap:10px"><button class="ghost" id="voiceBtn">🔊 Voice on</button><button class="gold" id="demoBtn">🎬 Auto demo with voice over</button></div>
</header>
<main>
 <aside><div class="card"><h3>Report an incident</h3>
  <label>Emergency type</label><select id="type"></select>
  <label>Where did it happen?</label><select id="loc"></select>
  <label>Severity: <b id="sevv">3</b> of 5</label><input type="range" id="sev" min="1" max="5" value="3">
  <label>Blocked junctions (tap to block)</label><div class="chips" id="blk"></div>
  <label>People inside each building</label><div class="occ" id="occ"></div>
  <div class="btns"><button class="primary" id="simBtn">🚨 Simulate emergency</button><button class="ghost" id="rstBtn">Reset</button></div>
 </div></aside>
 <section>
  <div class="kpis">
   <div class="kpi"><small>People on campus</small><div class="big" id="kTot">–</div></div>
   <div class="kpi"><small>Evacuated</small><div class="big" id="kEv">0</div><div class="bar"><i id="pb"></i></div></div>
   <div class="kpi"><small>Full evacuation time</small><div class="big" id="kEta">–</div></div>
   <div class="kpi"><small>Risk level</small><div class="big" id="kRisk">–</div><div class="bar"><i id="rb" style="background:linear-gradient(90deg,var(--am),var(--rd))"></i></div></div>
  </div>
  <div class="card"><svg id="map" viewBox="0 0 900 540"></svg>
   <div class="legend"><span><b style="background:var(--rd)"></b>Incident / blocked</span><span><b style="background:var(--cy)"></b>Safe route</span><span><b style="background:var(--gr)"></b>Assembly point</span><span><b style="background:#fff"></b>People moving</span></div></div>
  <div class="card"><h3>Evacuation plan</h3><table><thead><tr><th>Building</th><th>People</th><th>Goes to</th><th>Distance</th><th>Time</th><th>Status</th></tr></thead><tbody id="tbl"><tr><td colspan="6" style="color:var(--mut)">Choose an incident and press Simulate emergency.</td></tr></tbody></table></div>
 </section>
 <aside>
  <div class="card"><h3>Agent reasoning</h3><ol id="steps"><li style="color:var(--mut)">Waiting for an incident report.</li></ol></div>
  <div class="card"><h3>Assembly point load</h3><div id="asm"><div style="color:var(--mut);font-size:13px">No data yet.</div></div></div>
  <div class="card"><h3>Services and alerts</h3><div class="svc" id="svc"></div><div class="alert" id="alertBox" style="display:none"></div></div>
  <div class="card"><h3>Live log</h3><div id="log"></div></div>
 </aside>
</main>
<script>
const $=s=>document.querySelector(s), sleep=ms=>new Promise(r=>setTimeout(r,ms));
const ICON={MAIN:'🏛️',LIB:'📚',LAB:'💻',AUD:'🎭',CAF:'🍽️',HOS:'🏠'}, FX={fire:'🔥',quake:'🌍',gas:'☣️',flood:'🌊'};
let MAP,voiceOn=true,runId=0,demoOn=false;

function speak(t){return new Promise(res=>{
 if(!voiceOn||!window.speechSynthesis){res();return}
 const u=new SpeechSynthesisUtterance(t),vs=speechSynthesis.getVoices();
 u.voice=vs.find(v=>/en-IN/i.test(v.lang))||vs.find(v=>/^en/i.test(v.lang))||null;u.rate=1;
 u.onend=u.onerror=()=>res();speechSynthesis.speak(u)})}
function log(m){const t=new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
 $('#log').insertAdjacentHTML('afterbegin',`<div><b>${t}</b>  ${m}</div>`)}

async function init(){
 MAP=await (await fetch('/api/map')).json();
 const N=MAP.nodes,B=Object.keys(N).filter(k=>N[k].t==='b'),J=Object.keys(N).filter(k=>N[k].t==='j');
 $('#type').innerHTML=Object.entries(MAP.types).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');
 $('#loc').innerHTML=B.map(k=>`<option value="${k}">${N[k].name}</option>`).join('');
 $('#loc').value='LAB';
 $('#blk').innerHTML=J.map(k=>`<span class="chip" data-j="${k}" role="button" tabindex="0">${k}</span>`).join('');
 document.querySelectorAll('.chip').forEach(c=>{c.onclick=()=>c.classList.toggle('on');c.onkeydown=e=>{if(e.key==='Enter')c.click()}});
 $('#occ').innerHTML=B.map(k=>`<div><span>${N[k].name}</span><input type="number" min="0" max="2000" id="p_${k}" value="${N[k].pop}"></div>`).join('');
 $('#sev').oninput=e=>$('#sevv').textContent=e.target.value;
 $('#simBtn').onclick=()=>runSim();$('#rstBtn').onclick=reset;$('#demoBtn').onclick=demo;
 $('#voiceBtn').onclick=()=>{voiceOn=!voiceOn;$('#voiceBtn').textContent=voiceOn?'🔊 Voice on':'🔇 Voice off';if(!voiceOn)speechSynthesis.cancel()};
 reset();
}
function drawBase(){
 const N=MAP.nodes;let h=`<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0c1530"/><stop offset="1" stop-color="#080d20"/></linearGradient>
 <linearGradient id="bd" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2a3a7a"/><stop offset="1" stop-color="#1a2552"/></linearGradient></defs>
 <rect width="900" height="540" fill="url(#bg)"/>`;
 for(let x=0;x<=900;x+=45)h+=`<line x1="${x}" y1="0" x2="${x}" y2="540" stroke="#ffffff" stroke-opacity=".035"/>`;
 for(let y=0;y<=540;y+=45)h+=`<line x1="0" y1="${y}" x2="900" y2="${y}" stroke="#ffffff" stroke-opacity=".035"/>`;
 MAP.edges.forEach(([a,b])=>{h+=`<line x1="${N[a].x}" y1="${N[a].y}" x2="${N[b].x}" y2="${N[b].y}" stroke="#18224a" stroke-width="18" stroke-linecap="round"/>
 <line x1="${N[a].x}" y1="${N[a].y}" x2="${N[b].x}" y2="${N[b].y}" stroke="#3a4a86" stroke-width="1.5" stroke-dasharray="5 7"/>`});
 h+='<g id="routes"></g>';
 for(const[k,n]of Object.entries(N)){
  if(n.t==='b')h+=`<g id="n_${k}"><rect x="${n.x-58}" y="${n.y-25}" width="116" height="50" rx="13" fill="url(#bd)" stroke="#6b7fe0" stroke-opacity=".5"/>
   <text x="${n.x-40}" y="${n.y+7}" font-size="22" text-anchor="middle">${ICON[k]}</text>
   <text x="${n.x+8}" y="${n.y-3}" font-size="12.5" font-weight="700" fill="#e8edff" text-anchor="middle">${n.name}</text>
   <text x="${n.x+8}" y="${n.y+13}" font-size="11" fill="#8b97c4" text-anchor="middle" id="pc_${k}">${n.pop} people</text></g>`;
  else if(n.t==='j')h+=`<g id="n_${k}"><circle cx="${n.x}" cy="${n.y}" r="9" fill="#101a3d" stroke="#8b7bff" stroke-width="2"/><text x="${n.x}" y="${n.y-15}" font-size="11" fill="#8b97c4" text-anchor="middle">${k}</text></g>`;
  else h+=`<g><circle cx="${n.x}" cy="${n.y}" r="30" fill="rgba(52,224,161,.14)" stroke="#34e0a1" stroke-width="2"/><text x="${n.x}" y="${n.y+8}" font-size="24" text-anchor="middle">⛑️</text>
   <text x="${n.x}" y="${n.y+50}" font-size="12" font-weight="700" fill="#34e0a1" text-anchor="middle">${n.name}</text></g>`;
 }
 h+='<g id="fx"></g><g id="ppl"></g>';$('#map').innerHTML=h;
}
function reset(){runId++;speechSynthesis&&speechSynthesis.cancel();drawBase();
 ['#kTot','#kEta','#kRisk'].forEach(s=>$(s).textContent='–');$('#kEv').textContent='0';$('#pb').style.width='0';$('#rb').style.width='0';
 $('#kRisk').style.color='';$('#simBtn').disabled=false;
 $('#tbl').innerHTML='<tr><td colspan="6" style="color:var(--mut)">Choose an incident and press Simulate emergency.</td></tr>';
 $('#steps').innerHTML='<li style="color:var(--mut)">Waiting for an incident report.</li>';
 $('#asm').innerHTML='<div style="color:var(--mut);font-size:13px">No data yet.</div>';$('#svc').innerHTML='';$('#alertBox').style.display='none';
 $('#log').innerHTML='';log('Dashboard ready. Enter an incident to begin.')}

function setEvac(e,t){$('#kEv').textContent=Math.round(e);$('#pb').style.width=(t?100*e/t:0)+'%'}
function animate(R,id){return new Promise(resolve=>{
 const g=$('#ppl');g.innerHTML='';const items=[];let total=0;
 R.results.filter(r=>r.status==='evacuating'&&r.coords.length>1).forEach(r=>{
  const lens=[];let tot=0;for(let i=1;i<r.coords.length;i++){const l=Math.hypot(r.coords[i][0]-r.coords[i-1][0],r.coords[i][1]-r.coords[i-1][1]);lens.push(l);tot+=l}
  const n=Math.min(14,Math.max(3,Math.round(r.people/25))),dur=Math.max(5,Math.min(13,r.walk_s/7));total+=r.people;
  for(let i=0;i<n;i++){const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('r',5);c.setAttribute('fill',r.hazard?'#ffc247':'#fff');
   c.setAttribute('opacity',0);g.appendChild(c);items.push({c,pts:r.coords,lens,tot,dur,delay:i*.35+(r.hazard?0:.5),w:r.people/n,done:false})}});
 const t0=performance.now();
 (function frame(now){
  if(id!==runId){resolve();return}
  const t=(now-t0)/1000;let ev=0,all=true;
  items.forEach(o=>{
   const p=(t-o.delay)/o.dur;if(p<0){all=false;return}
   if(p>=1){ev+=o.w;o.c.setAttribute('opacity',0);return}
   all=false;let d=p*o.tot,i=0;while(i<o.lens.length-1&&d>o.lens[i]){d-=o.lens[i];i++}
   const f=d/o.lens[i],a=o.pts[i],b=o.pts[i+1];
   o.c.setAttribute('cx',a[0]+(b[0]-a[0])*f);o.c.setAttribute('cy',a[1]+(b[1]-a[1])*f);o.c.setAttribute('opacity',1)});
  setEvac(all?total:ev,total);
  all?resolve():requestAnimationFrame(frame)})(t0)})}

async function runSim(){
 const id=++runId;speechSynthesis.cancel();$('#simBtn').disabled=true;
 const people={};Object.keys(MAP.nodes).filter(k=>MAP.nodes[k].t==='b').forEach(k=>people[k]=+$('#p_'+k).value||0);
 const body={type:$('#type').value,location:$('#loc').value,severity:+$('#sev').value,people,
  blocked:[...document.querySelectorAll('.chip.on')].map(c=>c.dataset.j)};
 const R=await (await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
 drawBase();const N=MAP.nodes;
 R.results.forEach(r=>{const e=$('#pc_'+r.id);if(e)e.textContent=r.people+' people'});
 let fx='',rt='';
 R.results.filter(r=>r.coords.length>1).forEach(r=>{rt+=`<polyline class="flow" points="${r.coords.map(c=>c.join(',')).join(' ')}" fill="none" stroke="${r.hazard?'#ffc247':'#2dd4ee'}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" opacity=".85"/>`});
 $('#routes').innerHTML=rt;
 const hz=N[R.hazard];fx+=`<circle class="pulse" cx="${hz.x}" cy="${hz.y}" r="40" fill="rgba(255,77,109,.5)"/><text x="${hz.x+44}" y="${hz.y-26}" font-size="30" text-anchor="middle">${FX[R.type]}</text>
  <rect x="${hz.x-58}" y="${hz.y-25}" width="116" height="50" rx="13" fill="none" stroke="#ff4d6d" stroke-width="3"/>`;
 R.blocked.filter(k=>k!==R.hazard).forEach(k=>{fx+=`<circle cx="${N[k].x}" cy="${N[k].y}" r="14" fill="rgba(255,77,109,.25)" stroke="#ff4d6d" stroke-width="2"/><text x="${N[k].x}" y="${N[k].y+6}" font-size="17" font-weight="800" fill="#ff4d6d" text-anchor="middle">✕</text>`});
 R.results.filter(r=>r.status==='trapped').forEach(r=>{fx+=`<rect class="shake" x="${N[r.id].x-62}" y="${N[r.id].y-29}" width="124" height="58" rx="15" fill="none" stroke="#ff4d6d" stroke-width="3" stroke-dasharray="6 5"/>`});
 $('#fx').innerHTML=fx;
 $('#kTot').textContent=R.total;$('#kEta').textContent=R.max_eta+' s';$('#kEv').textContent='0';$('#pb').style.width='0';
 $('#kRisk').textContent=R.level;$('#kRisk').style.color=R.level==='Critical'?'var(--rd)':R.level==='High'?'var(--am)':'var(--gr)';$('#rb').style.width=R.score+'%';
 $('#tbl').innerHTML=R.results.map(r=>`<tr><td>${r.name}${r.hazard?' 🔥':''}</td><td>${r.people}</td><td>${r.assembly?N[r.assembly].name.split(' (')[0]:'–'}</td><td>${r.meters?r.meters+' m':'–'}</td><td>${r.eta_s?r.eta_s+' s':'–'}</td>
  <td><span class="tag ${r.status==='evacuating'?'t-ev':r.status==='trapped'?'t-tr':'t-em'}">${r.status==='trapped'?'Needs rescue':r.status==='empty'?'Empty':'Evacuating'}</span></td></tr>`).join('');
 $('#asm').innerHTML=R.loads.map(a=>`<div class="al"><div><span>${a.name}</span><span>${a.load} / ${a.cap}</span></div><div class="bar"><i style="width:${Math.min(100,100*a.load/a.cap)}%;background:${a.load>a.cap?'var(--rd)':''}"></i></div></div>`).join('');
 $('#svc').innerHTML=R.services.map(s=>`<span>${s}</span>`).join('');$('#alertBox').style.display='block';$('#alertBox').textContent='📢 '+R.alert;
 log(`<b>${R.label}</b> at ${N[R.hazard].name}, severity ${body.severity}. Risk ${R.level}.`);
 $('#steps').innerHTML='';speak(R.narration);
 (async()=>{for(const s of R.steps){if(id!==runId)return;$('#steps').insertAdjacentHTML('beforeend',`<li>${s}</li>`);log(s);await sleep(450)}})();
 await sleep(1000);await animate(R,id);
 if(id===runId){log(`Evacuation complete: ${R.safe} safe${R.trapped?', '+R.trapped+' waiting for rescue':''}.`);$('#simBtn').disabled=false}
 return R}

async function demo(){
 if(demoOn)return;demoOn=true;$('#demoBtn').disabled=true;reset();
 const set=(t,l,s,b)=>{$('#type').value=t;$('#loc').value=l;$('#sev').value=s;$('#sevv').textContent=s;
  document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on',b.includes(c.dataset.j)))};
 await speak('Welcome. This is the A I based Smart Campus Emergency Response and Evacuation Agent. It senses an incident, assesses the risk, plans the safest routes using Dijkstra\'s shortest path algorithm, balances the load on assembly points, and alerts emergency services, all in real time.');
 set('fire','LAB',4,['J4']);await speak('Emergency one. Fire. A fire breaks out in the C S Labs with severity four, and junction J4 is blocked by a vehicle. Watch the agent mark the danger zone, find the shortest safe routes, and send everyone to the assembly points.');await runSim();await sleep(800);
 set('quake','AUD',5,[]);await speak('Emergency two. Earthquake. A severe earthquake hits the Auditorium, the most crowded building on campus. The agent gives the Auditorium first priority and spreads people across both assembly points.');await runSim();await sleep(800);
 set('gas','CAF',3,['J1']);await speak('Emergency three. Gas leak. A leak in the Cafeteria while junction J1 is closed. The agent reroutes every group around the blocked road.');await runSim();await sleep(800);
 set('flood','HOS',4,['J2']);await speak('Emergency four. Flood. Water enters the Hostel, and junction J2 is closed. The agent finds new paths and alerts the disaster management team.');await runSim();
 await speak('That completes all four emergencies. In every case, the agent reacts in seconds, avoids danger zones, keeps assembly points within capacity, and alerts the right services. Thank you.');
 demoOn=false;$('#demoBtn').disabled=false}
speechSynthesis&&speechSynthesis.getVoices();init();
</script></body></html>
"""

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Smart Campus Emergency Agent running at http://localhost:{PORT}  (Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")