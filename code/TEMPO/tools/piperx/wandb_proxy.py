"""复用服务器已有代理，生成仅服务W&B的私有本机配置。"""
import os
from pathlib import Path
import shutil
import subprocess
import time

import requests
import yaml


def reachable(proxy):
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get('https://api.wandb.ai/graphql',
                               proxies={'http': proxy, 'https': proxy}, timeout=12)
        return response.status_code in (200, 400, 401, 405)
    except requests.RequestException:
        return False
    finally:
        session.close()


def start_proxy(settings):
    source = Path(settings['source_directory'])
    private = Path(settings['work_directory'])
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    config = yaml.safe_load((source/'config.yaml').read_text())
    groups = [group['name'] for group in config.get('proxy-groups', [])]
    if not groups:
        raise ValueError('existing proxy group missing')
    selected = groups[0]
    for rule in config.get('rules', []):
        parts = rule.split(',')
        if parts[0] in ('MATCH', 'FINAL') and parts[1] in groups:
            selected = parts[1]
    for key in ('port', 'socks-port', 'redir-port', 'tproxy-port', 'listeners',
                'external-controller', 'external-controller-tls', 'external-controller-unix',
                'external-controller-pipe', 'external-ui', 'secret', 'rule-providers', 'dns', 'tun', 'sniffer'):
        config.pop(key, None)
    config.update({'mixed-port': settings['port'], 'allow-lan': False,
                   'bind-address': '127.0.0.1', 'mode': 'rule', 'log-level': 'warning'})
    # 仅允许实验记录所需域名，节点连接凭据存入用户私有缓存目录。
    domains = ('wandb.ai', 'wandb.me', 'wandb.com', 'googleapis.com', 'amazonaws.com')
    config['rules'] = [f'DOMAIN-SUFFIX,{domain},{selected}' for domain in domains]+['MATCH,REJECT']
    for provider in config.get('proxy-providers', {}).values():
        if 'path' in provider:
            original = Path(provider['path'])
            original = original if original.is_absolute() else source/original
            target = private/'providers'/original.name
            target.parent.mkdir(exist_ok=True, mode=0o700)
            if original.is_file():
                shutil.copyfile(original, target)
                target.chmod(0o600)
            provider['path'] = str(target)
    for name in ('cache.db', 'Country.mmdb', 'geoip.dat', 'geosite.dat', 'ASN.mmdb'):
        if (source/name).is_file() and not (private/name).exists():
            shutil.copyfile(source/name, private/name)
            (private/name).chmod(0o600)
    path = private/'config.yaml'
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    path.chmod(0o600)
    log_path = private/'proxy.log'
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, 'a') as log:
        process = subprocess.Popen([str(source/'bin/mihomo'), '-d', str(private), '-f', str(path)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    endpoint = f'http://127.0.0.1:{settings["port"]}'
    time.sleep(2)
    if process.poll() is not None or not reachable(endpoint):
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        raise RuntimeError('native W&B proxy unavailable')
    return process, endpoint
