"""Credential-bearing API requests must not silently change destinations."""
import ipaddress
import urllib.error
import urllib.parse
import urllib.request


def validate_url(url):
    parts = urllib.parse.urlsplit(url)
    try:
        local = ipaddress.ip_address(parts.hostname or '').is_loopback
    except ValueError:
        local = parts.hostname == 'localhost'
    if not parts.hostname or parts.username is not None or parts.password is not None:
        raise ValueError('API URL должен содержать хост и не должен содержать логин или пароль.')
    if parts.scheme != 'https' and not (parts.scheme == 'http' and local):
        raise ValueError('Для удалённого API требуется HTTPS. HTTP разрешён только для localhost.')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'API redirects are disabled', headers, fp)


def open_api(request, timeout):
    validate_url(request.full_url if isinstance(request, urllib.request.Request) else request)
    return urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout)
