import json


if __name__ == '__main__':
    with open("/slot/sandbox/j/hosts.json", 'r') as f:
        hosts = json.load(f)

    for host in hosts:
        if host['role'] == 'master':
            master_addr = host['ip6']
            print(master_addr) # Used to assign the port via bash
