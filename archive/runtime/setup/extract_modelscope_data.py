"""校验ModelScope压缩包并原子发布数据集，避免训练读取半成品。"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

project = Path('/data-minio/taiyi/tempo')
archive = project/'data/downloads/piperx_rotating_red_square_tempo.zip'
staging = project/'data/.modelscope_extract'
with zipfile.ZipFile(archive) as package:
    failure = package.testzip()
    if failure:
        raise ValueError(f'压缩包校验失败 {failure}')
    staging.mkdir(parents=True,exist_ok=True)
    for member in package.infolist():
        (staging/member.filename).resolve().relative_to(staging.resolve())
    package.extractall(staging)
candidates = list(staging.rglob('meta/episodes.jsonl'))
assert len(candidates) == 1
fresh = candidates[0].parents[1]
episodes = [json.loads(line) for line in candidates[0].read_text().splitlines()]
assert len(episodes) == 393 and sum(row['length'] for row in episodes) == 44398
assert len(list(fresh.rglob('*.mp4'))) == 786
assert len(list(fresh.rglob('*.parquet'))) == 393
root = project/'data/data11_7/piperx_rotating_red_square'
root.parent.mkdir(parents=True,exist_ok=True)
if root.exists():
    previous = project/'runtime/setup/interrupted_dataset_transfer'
    root.rename(previous)
fresh.rename(root)
shutil.rmtree(staging)
# 已被完整下载替换的半成品仅属于本次配置，可以清除。
previous = project/'runtime/setup/interrupted_dataset_transfer'
if previous.exists():
    shutil.rmtree(previous)
digest=hashlib.sha256()
files=list(sorted(p for p in root.rglob('*') if p.is_file()))
for path in files:
    digest.update(str(path.relative_to(root)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
report={'source':'modelscope','dataset':'taiyigong3333/piperx_rotation_red_block',
        'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'files':len(files),'bytes':sum(p.stat().st_size for p in files),
        'sha256_inventory':digest.hexdigest(),'episodes':393,'frames':44398,'videos':786,
        'zip_crc_verified':True}
(project/'docs/dataset_modelscope.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
