import logging
from functools import lru_cache
from time import time_ns
from typing import List

import torch.cuda
from llama_cpp import LlamaTokenizer
from transformers import LlamaForCausalLM

from kohaku_bomber.prompts.dtg_local.generate import tag_gen
from kohaku_bomber.prompts.dtg_local.metainfo import SPECIAL, TARGET

_DEFAULT_SPECIAL_TAGS = ['1girl']
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_result(
        text_model: LlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        rating: str = "",
        artist: str = "",
        characters: str = "",
        copyrights: str = "",
        target: str = "long",
        special_tags: List[str] = None,
        general: str = "",
        aspect_ratio: float = 0.0,
        blacklist: str = "",
        escape_bracket: bool = False,
        temperature: float = 1.35,
) -> str:
    if special_tags is None:
        special_tags = _DEFAULT_SPECIAL_TAGS

    start = time_ns()
    print("=" * 50, "\n")
    # Use LLM to predict possible summary
    # This prompt allow model itself to make request longer based on what it learned
    # Which will be better for preference sim and pref-sum contrastive scorer
    prompt = f"""
rating: {rating or '<|empty|>'}
artist: {artist.strip() or '<|empty|>'}
characters: {characters.strip() or '<|empty|>'}
copyrights: {copyrights.strip() or '<|empty|>'}
aspect ratio: {f"{aspect_ratio:.1f}" or '<|empty|>'}
target: {'<|' + target + '|>' if target else '<|long|>'}
general: {", ".join(special_tags)}, {general.strip().strip(",")}<|input_end|>
""".strip()

    artist = artist.strip().strip(",").replace("_", " ")
    characters = characters.strip().strip(",").replace("_", " ")
    copyrights = copyrights.strip().strip(",").replace("_", " ")
    special_tags = [tag.strip().replace("_", " ") for tag in special_tags]
    general = general.strip().strip(",")
    black_list = set(
        [tag.strip().replace("_", " ") for tag in blacklist.strip().split(",")]
    )

    prompt_tags = special_tags + general.strip().strip(",").split(",")
    len_target = TARGET[target]
    llm_gen = ""

    for llm_gen, extra_tokens in tag_gen(
            text_model,
            tokenizer,
            prompt,
            prompt_tags,
            len_target,
            black_list,
            temperature=temperature,
            top_p=0.95,
            top_k=100,
            max_new_tokens=256,
            max_retry=5,
    ):
        logging.info(f'Yield LLM, cost time: {(time_ns() - start) / 1e9:.2f}s, length: {len(llm_gen)}')
        # yield "", llm_gen, f"Total cost time: {(time_ns() - start) / 1e9:.2f}s"

    general = f"{general.strip().strip(',')}, {','.join(extra_tokens)}"
    tags = general.strip().split(",")
    tags = [tag.strip() for tag in tags if tag.strip()]
    special = special_tags + [tag for tag in tags if tag in SPECIAL]
    tags = [tag for tag in tags if tag not in special]

    final_prompt = ", ".join(special)
    if characters:
        final_prompt += f", \n\n{characters}"
    if copyrights:
        final_prompt += ", "
        if not characters:
            final_prompt += "\n\n"
        final_prompt += copyrights
    if artist:
        final_prompt += f", \n\n{artist}"
    final_prompt += f""", \n\n{', '.join(tags)},
masterpiece, newest, absurdres, {rating}"""

    if escape_bracket:
        final_prompt = (
            final_prompt.replace("[", "\\[")
            .replace("]", "\\]")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    return final_prompt
    # yield final_prompt, llm_gen, f"Total cost time: {(time_ns() - start) / 1e9:.2f}s  |  Total general tags: {len(special + tags)}"


@lru_cache()
def _get_model(model_path: str):
    model = LlamaForCausalLM.from_pretrained(
        model_path, attn_implementation="flash_attention_2"
    ).requires_grad_(False).eval().half().to(_DEVICE)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_prompt_locally(
        rating: str,
        artist: str,
        characters: str,
        copyrights: str,
        target: str,
        special_tags: List[str],
        general: str,
        width: float,
        height: float,
        blacklist: str,
        escape_bracket: bool = False,
        temperature: float = 1.35,
):
    text_model, tokenizer = _get_model("KBlueLeaf/DanTagGen-beta")
    return get_result(
        text_model,
        tokenizer,
        rating,
        artist,
        characters,
        copyrights,
        target,
        special_tags,
        general,
        width / height,
        blacklist,
        escape_bracket,
        temperature,
    )
