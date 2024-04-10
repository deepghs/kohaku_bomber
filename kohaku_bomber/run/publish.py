import os
import shutil

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from imgutils.tagging import remove_underline
from tqdm import tqdm

from kohaku_bomber.source.base import gender_predict, list_character_tags
from .infer import infer_with_tag
from ..base import hf_client, hf_fs
from ..source import get_detailed_character_info

RUNS_DIR = os.path.abspath('runs')


def _init_repo(dst_repo):
    if not hf_client.repo_exists(repo_id=dst_repo, repo_type='dataset'):
        hf_client.create_repo(repo_id=dst_repo, repo_type='dataset', private=False)
        hf_client.update_repo_visibility(repo_id=dst_repo, repo_type='dataset', private=False)
        attr_lines = hf_fs.read_text(f'datasets/{dst_repo}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{dst_repo}/.gitattributes',
            os.linesep.join(attr_lines),
        )


def run_for_tag(tag):
    logging.info(f'Running for tag {tag!r} ...')
    info = get_detailed_character_info(tag)
    output_dir = os.path.join(RUNS_DIR, info['short_tag'])
    os.makedirs(output_dir, exist_ok=True)
    retval = infer_with_tag(tag, output_dir)
    return output_dir, {
        'tag': tag,
        'copyright': info['copyright'],
        'short_tag': info['short_tag'],
        'hprefix': info['hprefix'],
        'danbooru_id': info['id'],
        'danbooru_posts': info['post_count'],
        'gender': gender_predict(info['gender']),
        'core_tags': ', '.join(map(remove_underline, info['core_tags'])),
        **retval,
    }


def auto_sync(repository):
    _init_repo(repository)

    if hf_fs.exists(f'datasets/{repository}/metrics.csv'):
        df_m = pd.read_csv(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='metrics.csv',
        ))
        df_m = df_m.replace(np.NaN, None)
        exist_tags = set(df_m['tag'])
        records = df_m.to_dict('records')
    else:
        exist_tags = set()
        records = []

    all_tags = list(list_character_tags())
    all_tags = [tag for tag in all_tags if 'arknights' in tag]
    for tag in tqdm(all_tags):
        if tag in exist_tags:
            logging.info(f'Tag {tag!r} already crawled, skipped.')
            continue

        logging.info(f'Tag {tag!r} confirmed.')
        output_dir, row = run_for_tag(tag)
        records.append(row)

        with TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, row['hprefix']), exist_ok=True)
            shutil.copytree(output_dir, os.path.join(td, row['hprefix'], row['short_tag']))
            inner_readme_file = os.path.join(td, row['hprefix'], row['short_tag'], 'README.md')
            assert os.path.exists(inner_readme_file)

            df_m = pd.DataFrame(records)
            df_m.to_csv(os.path.join(td, 'metrics.csv'), index=False)

            r_records = []
            for item in records:
                r_records.append({
                    'ID': item['danbooru_id'],
                    'Tag': f'[{item["tag"]}]({item["hprefix"]}/{item["short_tag"]}/README.md)',
                    'Copyright': item['copyright'],
                    'Gender': item['gender'],
                    'Posts': item['danbooru_posts'],
                    'CCIP': item['ccip'],
                    'AIC': item['aic'],
                    'BP': item['bp'],
                    'Core Tags': item['core_tags'],
                })
            df_r = pd.DataFrame(r_records)
            df_r = df_r.sort_values(by=['ID'], ascending=[False])

            with open(os.path.join(td, 'README.md'), 'w') as f:
                print('---', file=f)
                print('license: mit', file=f)
                print('---', file=f)
                print('', file=f)

                print(f'# Test Index for Kohaku Delta', file=f)
                print(f'', file=f)

                print(df_r.to_markdown(index=False), file=f)
                print(f'', file=f)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Upload result for {tag!r}'
            )
            exist_tags.add(tag)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    auto_sync(
        repository=os.environ['DST_REPO']
    )
