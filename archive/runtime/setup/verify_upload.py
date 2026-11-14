"""发布获授权的本次训练配置，并回读检查gpu目录中的文件。"""
from pathlib import Path
import hashlib
import json
import shutil
from modelscope.hub.api import HubApi
p=Path('/data-minio/taiyi/tempo')
cpath=p/'configs/gpu_paper_chunk30_4gpu_20261114.json'; c=json.loads(cpath.read_text())
bundle=p/'runtime/setup/public_config'; bundle.mkdir(exist_ok=True)
shutil.copyfile(cpath,bundle/'controller.json')
shutil.copyfile(p/'docs/dataset_modelscope.json',bundle/'dataset_modelscope.json')
model_hashes={name:hashlib.sha256((p/'code/TEMPO'/name).read_bytes()).hexdigest() for name in
              ('src/openpi/models/pi0_config.py','src/openpi/models_pytorch/pi0_pytorch.py')}
(bundle/'official_model_sha256.json').write_text(json.dumps(model_hashes,indent=2))
api=HubApi(); api.login(Path(c['modelscope']['token_file']).read_text().strip())
prefix=c['modelscope']['path_prefix']+'/configs'
api.upload_folder(repo_id='taiyigong3333/tempo',repo_type='dataset',folder_path=bundle,path_in_repo=prefix,
                  max_workers=2,use_cache=True,disable_tqdm=True,sync_remote_repo=False,
                  tracker_path=p/'runtime/setup/config_upload_cache',commit_message='Publish GPU training configuration')
remote={row.path.lstrip('/'):row for row in api._api.list_repo_files('taiyigong3333/tempo','dataset',recursive=True)}
for path in bundle.iterdir():
    item=remote[prefix+'/'+path.name]
    assert item.size==path.stat().st_size
    if item.sha256: assert item.sha256.lower()==hashlib.sha256(path.read_bytes()).hexdigest()
result={'repo_id':'taiyigong3333/tempo','path_in_repo':prefix,'remote_files_verified':3}
(p/'docs/modelscope_upload_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
