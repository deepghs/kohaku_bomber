import glob
import logging
import os
import re

import pandas as pd
from imgutils.data import load_image
from imgutils.sd import parse_sdmeta_from_text, get_sdmeta_from_image
from natsort import natsorted
from sdeval.controllability import BikiniPlusMetrics
from sdeval.corrupt import AICorruptMetrics
from sdeval.fidelity import CCIPMetrics
from tqdm import tqdm


def _name_safe(name_text):
    return re.sub(r'[\W_]+', '_', name_text).strip('_')


def infer_with_tag(
        tag, output_dir: str,
        base_model_name: str = 'kohaku-xl-delta-rev1',
        sampler_name: str = 'DPM++ 2M SDE Heun',
        cfg_scale: float = 5.0,
        steps: int = 24,
        firstphase_width: int = 896,
        firstphase_height: int = 1344,
        hr_resize_x: int = 1344,
        hr_resize_y: int = 2016,
        denoising_strength: float = 0.4,
        hr_second_pass_steps: int = 16,
        hr_upscaler: str = 'Lanczos',
        seed: int = 114512,
        enable_hr: bool = True,
        clip_skip: int = 1,
        schedule_rho: float = 0.35,
):
    os.makedirs(output_dir, exist_ok=True)

    from ..source import iter_prompts_for_tag, get_np_feats
    from ..base import get_webui_client, auto_init_webui

    auto_init_webui()
    client = get_webui_client()

    client.util_set_model(base_model_name)
    painting_maps = {}

    titles = []
    for title, prompt, neg_prompt, bs in iter_prompts_for_tag(tag):
        logging.info(f'Inferring for {title!r} ...')
        result = client.txt2img(
            prompt=prompt,
            negative_prompt=neg_prompt,
            batch_size=bs,
            sampler_name=sampler_name,
            cfg_scale=cfg_scale,
            steps=steps,
            firstphase_width=firstphase_width,
            firstphase_height=firstphase_height,
            hr_resize_x=hr_resize_x,
            hr_resize_y=hr_resize_y,
            denoising_strength=denoising_strength,
            hr_second_pass_steps=hr_second_pass_steps,
            hr_upscaler=hr_upscaler,
            seed=seed,
            enable_hr=enable_hr,
            override_settings={
                'CLIP_stop_at_last_layers': clip_skip,
                'rho': schedule_rho,
                'always_discard_next_to_last_sigma': True,
                'emphasis': 'No norm',
                'eta_ancestral': 0.35,
                'k_sched_type': 'polyexponential',
            },
        )

        logging.info('Saving images ...')
        titles.append(title)
        painting_maps[title] = []
        for i, image in enumerate(tqdm(result.images)):
            param_text = image.info.get('parameters')
            sdmeta = parse_sdmeta_from_text(param_text)
            name = f'{_name_safe(title)}_{i}.png'
            dst_image_file = os.path.join(output_dir, name)
            image.save(dst_image_file, pnginfo=sdmeta.pnginfo)
            painting_maps[title].append(name)

    ccip_metrics = CCIPMetrics(None, feats=get_np_feats(tag))
    bp_metrics = BikiniPlusMetrics()
    aic_metrics = AICorruptMetrics()

    png_files = natsorted(glob.glob(os.path.join(output_dir, '*.png')))
    png_filenames = [os.path.relpath(f, output_dir) for f in png_files]
    ccip_score_seq = ccip_metrics.score(png_files, mode='seq')
    ccip_score = ccip_score_seq.mean().item()
    aic_score_seq = aic_metrics.score(png_files, mode='seq')
    aic_score = aic_score_seq.mean().item()
    bp_score_seq = bp_metrics.score(png_files, mode='seq')
    bp_score = bp_score_seq.mean().item()

    df = pd.DataFrame({
        'filename': png_filenames,
        'ccip': ccip_score_seq,
        'aic': aic_score_seq,
        'bp': bp_score_seq,
    })
    df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    dx = {item['filename']: item for item in df.to_dict('records')}

    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        print(f'# {tag}', file=f)
        print(f'', file=f)

        print(f'## Integration', file=f)
        print(f'', file=f)
        print(f'Average CCIP: {ccip_score:.4f}, AIC: {aic_score:.4f}, BP: {bp_score:.4f}', file=f)
        print(f'', file=f)
        print(df.to_markdown(index=False), file=f)
        print(f'', file=f)

        print(f'## Samples', file=f)
        print(f'', file=f)

        for title in titles:
            print(f'### {title}', file=f)
            print(f'', file=f)

            seeds = [
                int(get_sdmeta_from_image(os.path.join(output_dir, filename)).parameters['Seed'])
                for filename in painting_maps[title]
            ]
            columns = ['Seeds', *map(str, seeds)]
            rows = [
                ('Images', *(f'![{filename}]({filename})' for filename in painting_maps[title])),
                ('Files', *painting_maps[title]),
                ('CCIP', *(f'{dx[filename]["ccip"]:.4f}' for filename in painting_maps[title])),
                ('AIC', *(f'{dx[filename]["aic"]:.4f}' for filename in painting_maps[title])),
                ('BP', *(f'{dx[filename]["bp"]:.4f}' for filename in painting_maps[title])),
            ]
            df_t = pd.DataFrame(columns=columns, data=rows)

            print(df_t.to_markdown(index=False), file=f)
            print(f'', file=f)

            first_file = os.path.join(output_dir, painting_maps[title][0])
            image = load_image(first_file, mode=None, force_background=None)
            if image.info.get('parameters'):
                print(f'Meta information of `{title}`:', file=f)
                print(f'', file=f)
                print(f'```', file=f)
                print(image.info.get('parameters'), file=f)
                print(f'```', file=f)
                print(f'', file=f)

    return {
        'tag': tag,
        'ccip': ccip_score,
        'aic': aic_score,
        'bp': bp_score,
    }
