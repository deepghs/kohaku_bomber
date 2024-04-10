from functools import lru_cache
from typing import Literal, List

from gradio_client import Client
from imgutils.tagging import remove_underline


@lru_cache()
def _dtg_online_client():
    client = Client("KBlueLeaf/DTG-demo")
    return client


RatingTyping = Literal['safe', 'sensitive', 'nsfw', 'nsfw, explicit']
LengthTyping = Literal['very_short', 'short', 'long', 'very_long']
TypesTyping = List[Literal[
    '1girl', '2girls', '3girls', '4girls', '5girls', '6+girls', 'multiple_girls',
    '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys', 'multiple_boys',
    'male_focus', '1other', '2others', '3others', '4others', '5others', '6+others', 'multiple_others'
]]


def get_dtg_prompt(character: str, copyright: str, gen_type: TypesTyping, core_tags: List[str],
                   prompt_tags: List[str], width: int = 1344, height: int = 2016, escape: bool = True,
                   rating: RatingTyping = 'safe', length: LengthTyping = 'long', temperature: float = 1.35):
    client = _dtg_online_client()

    result = client.predict(
        "KBlueLeaf/DanTagGen-beta",
        # Literal['KBlueLeaf/DanTagGen-alpha', 'KBlueLeaf/DanTagGen-beta'] in 'Model' Dropdown component
        rating,  # Literal['safe', 'sensitive', 'nsfw', 'nsfw, explicit'] in 'Rating' Radio component
        "",  # str in 'Artist' Textbox component
        character,  # str in 'Characters' Textbox component
        copyright,  # str in 'Copyrights(Series)' Textbox component
        length,  # Literal['very_short', 'short', 'long', 'very_long'] in 'Target length' Radio component
        gen_type,
        # List[Literal['1girl', '2girls', '3girls', '4girls', '5girls', '6+girls', 'multiple_girls', '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys', 'multiple_boys', 'male_focus', '1other', '2others', '3others', '4others', '5others', '6+others', 'multiple_others']] in 'Special tags' Dropdown component
        f"""
        {', '.join(map(remove_underline, core_tags))},
        
        {', '.join(map(remove_underline, prompt_tags))},
        """,  # str in 'Input your general tags' Textbox component

        width,  # float (numeric value between 256 and 4096) in 'Width' Slider component
        height,  # float (numeric value between 256 and 4096) in 'Height' Slider component
        "",  # str in 'tag Black list (seperated by comma)' Textbox component
        escape,  # bool in 'Escape bracket' Checkbox component
        temperature,  # float (numeric value between 0.1 and 2) in 'Temperature' Slider component
        api_name="/wrapper"
    )
    return result[0]
