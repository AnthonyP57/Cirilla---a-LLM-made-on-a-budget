import fandom
import json
import os
from tqdm import tqdm

fandom.set_wiki('Witcher')
fandom.set_lang('en')

urls=[]
n_files = len(os.listdir('./witcher_json'))
preprocessed = [i.split('.')[0] for i in os.listdir('./witcher_fandom')]
for i, file in enumerate(os.listdir('./witcher_json')):
    with open(f'./witcher_json/{file}', 'r') as f:
        data = json.load(f)
        iterator = tqdm(data, desc=f'File: {i+1} / {n_files}')
        for d in iterator:
            try:
                info = fandom.search(d)
                info = info[0][0]
                info = fandom.page(info)
                url = info.url
                if url not in urls:
                    title = info.title
                    if title not in preprocessed:
                        text = info.plain_text
                        with open(f'./witcher_fandom/{title}.txt', 'w') as f:
                            f.write(text)
                        urls.append(url)
                        iterator.write(title)
            except:
                iterator.write(f'Error: {url}')