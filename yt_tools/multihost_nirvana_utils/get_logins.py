import json


if __name__ == '__main__':
    with open("/slot/sandbox/j/hosts.json", 'r') as f:
        hosts = json.load(f)

    username_at_hostnames = []
    for host in hosts:
        if host['role'] == 'slave':
             username_at_hostnames.append(
                 host["login"] + "@" + host['hostName']
             )
            
    print(" ".join(username_at_hostnames)) # Used to assign the port via bash
