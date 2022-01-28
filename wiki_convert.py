from pathlib import Path
import json
from tqdm import tqdm

folder = '/media/tao/_dde_data/Datasets/wiki_zh_2019/wiki_zh'
glob = '**/wiki*'
paths = sorted([*Path(folder).glob(glob)])

with open('./test_data/wiki_zh_2019.txt', 'w') as output_data:
    for path in tqdm(paths):
        with open(path, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                output_data.write(l['text'].strip())
