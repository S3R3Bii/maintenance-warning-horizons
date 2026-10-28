"""Retrieve an officially linked simulation archive without redistributing its members."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
URL='https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip'
EXPECTED='c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2'

def main():
    data=ROOT/'data';cache=data/'cache';raw=data/'raw';cache.mkdir(parents=True,exist_ok=True);raw.mkdir(exist_ok=True)
    archive=cache/'official-mirror.zip'
    if not archive.exists():
        with urllib.request.urlopen(URL,timeout=90) as response:archive.write_bytes(response.read())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED,'Changed source archive requires review'
    records=[]
    with zipfile.ZipFile(archive) as outer:
        member=next(n for n in outer.namelist() if n.endswith('CMAPSSData.zip'))
        with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
            for name in ['train_FD001.txt','test_FD001.txt','RUL_FD001.txt','test_FD002.txt','RUL_FD002.txt']:
                blob=inner.read(name);(raw/name).write_bytes(blob)
                records.append(dict(file='data/raw/'+name,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    manifest=dict(source_url=URL,archive_sha256=EXPECTED,retrieved_utc=datetime.now(timezone.utc).isoformat(),source_kind='NASA simulation benchmark, not measured commercial operations',redistribution='Raw members/original paper excluded; source access via official mirror',files=records)
    target=data/'manifest.json'
    if target.exists():
        old=json.loads(target.read_text());assert old['files']==records and old['archive_sha256']==EXPECTED
    else:target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'selected_members':len(records),'archive_sha256':EXPECTED,'raw_files_kept_local':True}))

if __name__=='__main__':main()
