import sys
from pathlib import Path
ROOT=Path("."); sys.path.insert(0,str(ROOT/"py"))
import ga_edge_closure_gnn_policy as GA
SUMO="sumo"
I={"A":("netA/sim_3od_x1.0.sumocfg","netA/network_v2.net.xml",1000,"E28 E40"),
   "B":("netB/sim_3od_x1.0.sumocfg","netB/netB.net.xml",2000,"E35 -E41 E38 E44 E20 E12"),
   "J":("netJ/wildau_6od_x1.0.sumocfg","netJ/Netzmodell2.net.xml",1500,"")}
for k,(cfg,net,dem,pr) in I.items():
    protect=set(pr.split()); eps=GA._extract_route_endpoints(str(ROOT/cfg))
    if eps: protect.update(eps)
    _c,um,_n=GA.load_candidates(str(ROOT/net),protect,None)
    d=Path(sys.argv[1])/f"base_{k}"; d.mkdir(parents=True,exist_ok=True)
    add=d/"none.add.xml"; GA.write_closures_additional(add,[],um,begin=0,end=86400)
    att=GA.run_single_sumo_and_score(sumo_bin=SUMO,sumocfg=str(ROOT/cfg),additional=add,
        outdir=d,reroute_period=30,time_to_teleport=-1,time_to_teleport_highways=-1,
        emit_tripinfo=False,demand_size=dem)
    print(f"BASELINE {k} (no closure, eval seed) ATT = {att:.2f}")
