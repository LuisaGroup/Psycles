#!/usr/bin/env python3
import re,json,subprocess,collections,pathlib
BASE=pathlib.Path(__file__).resolve().parent
CYCLESDIS='/var/tmp/psycles-lamp-routing-sCLxKs/barber-isa-47Dvhd/cycles-compute-isa.txt'
CYCLESCO='/var/tmp/psycles-lamp-routing-sCLxKs/barber-isa-47Dvhd/cycles-compute.co'
GENERICDIS='/var/tmp/psycles-isa-cause-audit-20260911/packed-opt-full/isa/hip_isa_12.dis'
GENERICCO='/var/tmp/psycles-isa-cause-audit-20260911/packed-opt-full/isa/hip_isa_12.co'
CANDDIS='/var/tmp/psycles-noise-shapes-20260911/candidate-1/isa/surface.dis'
CANDCO='/var/tmp/psycles-noise-shapes-20260911/candidate-1/isa/hip_isa_12.co'
HEADER=re.compile(r'^([0-9A-Fa-f]+) <([^>]+)>:')
ADDR=re.compile(r'//\s*([0-9A-Fa-f]+):')
OP=re.compile(r'\s*([A-Za-z_][A-Za-z0-9_.]*)\b')
class Module:
 def __init__(self,dis,co):
  self.dis,self.co=dis,co; self.syms={}; self.rows=collections.defaultdict(list);self.pad=collections.Counter()
  for l in subprocess.check_output(['readelf','-sW',co],text=True).splitlines():
   m=re.match(r'\s*\d+:\s+([0-9A-Fa-f]+)\s+(\S+)\s+FUNC\s+\S+\s+\S+\s+(\S+)\s+(.+)',l)
   if m and m.group(3) not in ('UND','ABS'):
    self.syms[m.group(4).strip()]=(int(m.group(1),16),int(m.group(2),0))
  self.starts={a:n for n,(a,z) in self.syms.items()};active=None
  for l in open(dis,errors='replace'):
   m=HEADER.match(l)
   if m:active=m.group(2);continue
   if active is None:continue
   m=ADDR.search(l)
   if not m:continue
   a=int(m.group(1),16);lo,z=self.syms.get(active,(0,0))
   if not (lo<=a<lo+z):self.pad[active]+=1;continue
   lane=[]
   for x in l.split('//',1)[0].split('::'):
    m=OP.match(x)
    if m:lane.append(m.group(1))
   self.rows[active].append((a,l.rstrip(),lane))
 def calls(self,name):
  # PC saved by s_getpc is the address of the next instruction, i.e. PC+4.
  # Relocated function offsets are signed 32-bit add immediates. Exact symbol
  # starts are mandatory; no nearest-symbol lookup or -4 correction is used.
  last={};out=[]
  for a,l,ops in self.rows[name]:
   m=re.search(r's_getpc_b64 s\[(\d+):(\d+)\]',l)
   if m:last[(int(m.group(1)),int(m.group(2)))]=(a,None)
   m=re.search(r's_add_co_u32 s(\d+), s\1, (0x[0-9A-Fa-f]+)',l)
   if m:
    reg=int(m.group(1));imm=int(m.group(2),16);imm-=1<<32 if imm>>31 else 0
    for k,(pc,old) in list(last.items()):
     if k[0]==reg and 0<=a-pc<100:last[k]=(pc,imm)
   m=re.search(r's_swappc_b64 s\[[^]]+\], s\[(\d+):(\d+)\]',l)
   if m:
    k=(int(m.group(1)),int(m.group(2)));v=last.get(k)
    t=(v[0]+4+v[1])&((1<<64)-1) if v and v[1] is not None else None
    out.append({'site':hex(a),'getpc':hex(v[0]) if v else None,'offset':v[1] if v else None,'target':hex(t) if t is not None else None,'symbol':self.starts.get(t)})
  return out
 def closure(self,root):
  seen=set();q=[root];edges={}
  while q:
   n=q.pop()
   if n in seen:continue
   seen.add(n);edges[n]=self.calls(n)
   q.extend(c['symbol'] for c in edges[n] if c['symbol'] and c['symbol'] not in seen)
  return seen,edges
 def profile(self,names):
  names=set(names);raw=collections.Counter(op for n in names for _,_,ops in self.rows[n] for op in ops)
  norm=collections.Counter()
  for op,c in raw.items():norm[re.sub(r'_e(?:32|64)$','',op.replace('v_dual_','v_'))]+=c
  def cnt(rx):return sum(c for op,c in norm.items() if re.search(rx,op))
  return {'symbols':len(names),'symbol_bytes':sum(self.syms[n][1] for n in names),'rows':sum(len(self.rows[n]) for n in names),'lanes':sum(raw.values()),'excluded_padding_rows':sum(self.pad[n] for n in names),'categories':{'branches':cnt(r'^(s_cbranch|s_branch|v_cbranch)'),'cndmask':cnt('cndmask'),'compares':cnt(r'^(v_cmp|s_cmp)'),'wait_delay':cnt(r'^s_(wait|delay)'),'alu_wait_delay':cnt(r'^s_(wait_alu|delay_alu)'),'memory_waits':cnt(r'^s_wait_(load|store|sample|bvh|km|ds|exp)'), 'scratch':cnt('scratch'),'global':cnt('global'),'flat':cnt('flat'),'image_sample':cnt(r'^image_sample'),'v_integer_hash':cnt(r'^v_(xor|and|or|not|alignbit|lshl|lshr|ashr|bfe|bfi|bit|mul_lo_u32|mul_hi_u32|add.*[ui]32|sub.*[ui]32)'),'f64':cnt('f64')},'raw_opcodes':dict(sorted(raw.items())),'normalized_opcodes':dict(sorted(norm.items()))}
 def graph_report(self,root):
  seen,edges=self.closure(root)
  return {'root':root,'entry':self.profile([root]),'inclusive_unique':self.profile(seen),'call_sites':sum(len(v) for v in edges.values()),'unresolved':[dict(c,caller=n) for n,v in edges.items() for c in v if c['symbol'] is None],'edges':edges,'symbols':{n:{'start':hex(self.syms[n][0]),'bytes':self.syms[n][1],'lanes':self.profile([n])['lanes']} for n in sorted(seen)}}
cycles=Module(CYCLESDIS,CYCLESCO);generic=Module(GENERICDIS,GENERICCO);candidate=Module(CANDDIS,CANDCO)
main='_Z17integrate_surfaceILj1979EEiPK16KernelGlobalsGPUiPf';noise='_Z18svm_node_tex_noisePfRK15SVMNodeTexNoise';fbm3='_Z9noise_fbm15HIP_vector_typeIfLj3EEfffb'
genericmain=next(n for n in generic.syms if n.startswith('kernel_') and not n.endswith('.kd'));candmain=next(n for n in candidate.syms if n.startswith('kernel_') and not n.endswith('.kd'))
report={'method':'Static decoded lanes; dual :: lanes counted separately; explicit ELF symbol-size bounds exclude padding; s_swappc targets require exact PC+4+signed immediate symbol start. Unique closure counts each symbol once, without dynamic or callsite frequency weighting.','artifacts':{'cycles_dis':CYCLESDIS,'cycles_co':CYCLESCO,'generic_dis':GENERICDIS,'generic_co':GENERICCO,'candidate_dis':CANDDIS,'candidate_co':CANDCO},'cycles':cycles.graph_report(main),'cycles_noise':cycles.graph_report(noise),'cycles_fbm3':cycles.graph_report(fbm3),'psycles_generic':generic.graph_report(genericmain),'psycles_candidate':candidate.graph_report(candmain)}
json.dump(report,open(BASE/'isa-report.json','w'),indent=2)
for label in ['cycles','cycles_noise','cycles_fbm3','psycles_generic','psycles_candidate']:
 r=report[label]
 print(label,'entry',r['entry']['lanes'],'inclusive',r['inclusive_unique']['lanes'],'symbols',r['inclusive_unique']['symbols'],'calls',r['call_sites'],'unresolved',len(r['unresolved']),'bytes',r['inclusive_unique']['symbol_bytes'])
 print(' categories',r['inclusive_unique']['categories'])
 print(' f64', {k:v for k,v in r['inclusive_unique']['normalized_opcodes'].items() if 'f64' in k or 'frexp' in k})
