import json
import socket


if __name__ == '__main__':
    my_host_name = socket.gethostname()
    with open("/slot/sandbox/j/hosts.json", 'r') as f:
        hosts = json.load(f)
    
    hosts_by_name = {host["hostName"]: host for host in hosts}
    assert my_host_name in hosts_by_name, (my_host_name, hosts_by_name)

    host = hosts_by_name[my_host_name]
    if host['role'] == 'master':
        print(0) # Used to assign the port via bash
    else:
        machine_rank = int(host["jobUid"].split('-')[-1])
        print(machine_rank + 1) # Used to assign the port via bash
