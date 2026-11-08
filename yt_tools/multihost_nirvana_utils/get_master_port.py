import json


if __name__ == '__main__':
    with open("/slot/sandbox/j/hosts.json", 'r') as f:
        hosts = json.load(f)

    for host in hosts:
        if host['role'] == 'master':
            for port in host['ports']:
                if port['name'] == "btl_port":
                    print(port['number']) # Used to assign the port via bash
