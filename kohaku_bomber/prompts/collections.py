import logging
from typing import List

from imgutils.tagging import remove_underline
from tqdm import tqdm

from .dtg import get_dtg_prompt

default_neg_tags = [
    "low quality", "worst quality", "normal quality", "text", "signature", "jpeg artifacts", "bad anatomy", "old",
    "early", "chibi"
]

generic_prompts = [
    # generic
    ('free (short, safe)', 4, 'short', 'safe', [], ["nsfw"]),
    ('free (short, sensitive)', 4, 'short', 'sensitive', [], []),
    ('free (long, safe)', 4, 'long', 'safe', [], ["nsfw"]),
    ('free (long, sensitive)', 4, 'long', 'sensitive', [], []),
    ('angry', 4, 'long', 'safe', ['angry', 'annoyed', 'portrait', 'looking at viewer'], []),

    # postures
    ('profile', 4, 'long', 'safe', ['profile', 'from side', 'upper body'], []),
    ('sit', 4, 'long', 'safe', ['sitting', 'sitting on chair', 'chair', 'cowboy shot', 'looking at viewer'], []),
    ('squat', 4, 'long', 'safe', ['squatting', 'cowboy shot', 'looking at viewer'], []),
]

boy_prompts = [
    *generic_prompts,

    # outfits
    ('short', 4, 'long', 'sensitive', ['topless male', 'shorts', "seaside", "night", "stary sky"], []),
    ('uniform', 4, 'long', 'safe', ['military uniform', 'handsome'], ["nsfw"]),
    ('suit', 4, 'long', 'safe', [
        'black business suit', 'tie', 'sunglasses', 'white gloves', 'white shirt', 'smoking', 'handsome'], ["nsfw"]),
    ('jacket', 4, 'long', 'safe', [
        'black jacket', 'white hoodie', 'open jacket', 'open clothes', 'hoodie', 'upper body',
    ], ["nsfw"]),

    # nsfws
    ('nude (standing)', 4, 'long', 'nsfw, explicit', [
        'standing', 'nude', 'completely nude', 'male focus', 'nipples', 'penis', 'erection',
        'looking at viewer', 'embarrassed', 'uncensored',
    ], []),
]

girl_prompts = [
    *generic_prompts,

    # outfits
    ('bikini', 4, 'long', 'sensitive', ["bikini", "seaside", "night", "stary sky"], []),
    ('maid', 4, 'long', 'safe', ["maid", "long maid dress"], ["nsfw"]),
    ('yukata', 4, 'long', 'safe', ['yukata', 'kimono'], ["nsfw"]),
    ('china', 4, 'long', 'sensitive', [
        'bare shoulders', 'bead bracelet', 'beads', 'china dress', 'chinese clothes', 'dress',
        'jewelry', 'looking at viewer', 'sleeveless', 'sleeveless dress', 'bracelet', 'smile', 'medium breasts',
        'thighs', 'cowboy shot', ':d', 'open mouth'
    ], []),

    # nsfws
    ('nude (lying)', 4, 'long', 'nsfw, explicit', [
        'lying on bed', 'nude', 'spread legs', 'arms up', 'mature', 'nipples',
        'pussy', 'pussy juice', 'looking at viewer', 'embarrassed', 'endured face', 'feet out of frame',
        'uncensored',
    ], []),
    ('nude (standing)', 4, 'long', 'nsfw, explicit', [
        'standing', 'nude', 'completely nude', 'mature', 'nipples', 'pussy', 'pussy juice',
        'looking at viewer', 'embarrassed', 'uncensored',
    ], []),
    ('nude (tentacle)', 4, 'long', 'nsfw, explicit', [
        'nude', 'nipples', 'tentacles', 'tentacle sex', 'tentacle grab',
        'pussy', 'pussy juice', 'embarrassed', 'uncensored',
    ], []),
]


def iter_prompt_pairs(character: str, copyright: str, gender: str = 'female', core_tags: List[str] = None,
                      width: int = 1344, height: int = 2016, temperature: float = 1.35):
    core_tags = list(core_tags or [])
    logging.info(f'Character {character!r}, copyright: {copyright!r}, gender: {gender!r}, core tags: {core_tags!r} ...')
    if gender == 'female':
        lst = girl_prompts
        gen_type = ['1girl']
    elif gender == 'male':
        lst = boy_prompts
        gen_type = ['1boy']
    else:
        lst = generic_prompts
        gen_type = []

    lst = lst[:1]
    for title, batch_size, length, rating, prompt_tags, neg_tags in tqdm(lst):
        logging.info(f'Generating prompts for {title!r} of character {character!r} ...')
        prompt = get_dtg_prompt(
            character=character,
            copyright=copyright,
            length=length,
            rating=rating,
            gen_type=gen_type,
            core_tags=core_tags,
            prompt_tags=prompt_tags,
            width=width,
            height=height,
            temperature=temperature,
        )
        neg_prompt = ', '.join(map(remove_underline, [*default_neg_tags, *neg_tags]))
        yield title, prompt, neg_prompt, batch_size
