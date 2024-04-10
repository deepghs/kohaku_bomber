import logging
import os
from functools import lru_cache
from typing import Optional, Set

from hbutils.system import urlsplit
from webuiapi import WebUIApi

_WEBUI_CLIENT: Optional[WebUIApi] = None


def set_webui_server(host="127.0.0.1", port=7860, baseurl=None, use_https=False, **kwargs):
    global _WEBUI_CLIENT
    logging.info(f'Set webui server {"https" if use_https else "http"}://{host}:{port}/{baseurl or ""}')
    _WEBUI_CLIENT = WebUIApi(
        host=host,
        port=port,
        baseurl=baseurl,
        use_https=use_https,
        **kwargs
    )
    _get_client_scripts.cache_clear()


def get_webui_client() -> WebUIApi:
    if _WEBUI_CLIENT:
        return _WEBUI_CLIENT
    else:
        raise OSError('Webui server not set, please set that with `set_webui_server` function.')


@lru_cache()
def _get_client_scripts() -> Set[str]:
    client = get_webui_client()
    scripts = client.get_scripts()['txt2img']
    return set(map(str.lower, scripts))


def _set_webui_server_from_env():
    webui_server = os.environ.get('CH_WEBUI_SERVER')
    if webui_server:
        url = urlsplit(webui_server)
        if ':' in url.host:
            host, port = url.host.split(':', maxsplit=1)
            port = int(port)
        else:
            host, port = url.host, 80

        set_webui_server(
            host=host,
            port=port,
            use_https=url.scheme == 'https',
        )
    else:
        logging.info('No webui server settings found.')


def auto_init_webui():
    if not _WEBUI_CLIENT:
        _set_webui_server_from_env()
