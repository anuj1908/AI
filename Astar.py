#A*
import queue as Q
from RMP import dict_gn
from RMPhn import dict_hn
start="Arad"
goal="Bucharest"
result=""
def get_fn(citystr):
    cities=citystr.split(",")
    hn=gn=0
    for ctr in range(0,len(cities)-1):
        gn=gn+dict_gn[cities[ctr]][cities[ctr+1]]
    hn=dict_hn[cities[len(cities)-1]]
    return(hn+gn)
def expand(cityq):
    global result
    tot,citystr,thiscity=cityq.get()
    if thiscity==goal:
        result=citystr+"::"+str(tot)
        return
    for city in dict_gn[thiscity]:
        cityq.put((get_fn(citystr+","+city),citystr+","+city,city))
    expand(cityq)
def main():
    cityq=Q.PriorityQueue()
    thiscity=start
    cityq.put((get_fn(start),start,thiscity))
    expand(cityq)
    print("The A* path with the total is :")
    print(result)
main()
