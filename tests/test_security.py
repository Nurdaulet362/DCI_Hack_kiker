import os
import stat
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cqa.network import open_api, validate_url
from cqa.settings import ROOT
from test_http import HttpTests


class SecurityHTTPTests(HttpTests):
    def test_secret_paths_are_not_served(self):
        for path in ['/.env', '/data/cqa.sqlite3', '/data/dev.json', '/../.env', '/%2e%2e/.env']:
            with self.assertRaises(urllib.error.HTTPError) as err:
                self.call(path)
            self.assertEqual(err.exception.code, 404)
            err.exception.close()

    def test_dns_rebinding_host_is_rejected(self):
        req = urllib.request.Request(self.url+'/api/state', headers={'Host': 'attacker.example'})
        with self.assertRaises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        self.assertEqual(err.exception.code, 403)
        err.exception.close()

    def test_fetch_metadata_blocks_cross_site_without_origin(self):
        req = urllib.request.Request(self.url+'/api/import', b'[]',
            {'Content-Type': 'application/json', 'Sec-Fetch-Site': 'cross-site'})
        with self.assertRaises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        self.assertEqual(err.exception.code, 403)
        err.exception.close()

    def test_sql_payload_is_literal_id(self):
        from test_core import conversation
        c = conversation(); c['conversation_id'] = "x'; DROP TABLE conversations; --"
        self.call('/api/import', [c])
        self.assertEqual(len(self.call('/api/state')['conversations']), 2)

    def test_database_permissions(self):
        if os.name != 'posix': self.skipTest('POSIX permissions only')
        self.assertEqual(stat.S_IMODE(os.stat(self.store.path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE((ROOT/'data').stat().st_mode), 0o700)


class OutboundSecurityTests(unittest.TestCase):
    def test_only_https_or_literal_loopback_http(self):
        for url in ['https://example.com/v1', 'http://localhost:11434', 'http://127.0.0.1:1', 'http://[::1]:1']:
            validate_url(url)
        for url in ['http://example.com/v1', 'file:///etc/passwd', 'ftp://example.com',
                    'https://user:password@example.com', 'http://127.0.0.1.attacker.example']:
            with self.assertRaises(ValueError): validate_url(url)

    def test_redirect_cannot_forward_authorization(self):
        received = []
        class Sink(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(self.headers.get('Authorization'))
                self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
            def log_message(self, *args): pass
        sink = ThreadingHTTPServer(('127.0.0.1', 0), Sink)
        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header('Location', f'http://127.0.0.1:{sink.server_port}/')
                self.end_headers()
            def log_message(self, *args): pass
        redirect = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
        for server in [sink, redirect]:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            request = urllib.request.Request(f'http://127.0.0.1:{redirect.server_port}/', b'{}',
                                             {'Authorization': 'Bearer dummy-test-only'})
            with self.assertRaises(urllib.error.HTTPError) as err:
                open_api(request, timeout=2)
            self.assertEqual(err.exception.code, 302)
            err.exception.close()
            self.assertEqual(received, [])
        finally:
            for server in [sink, redirect]: server.shutdown(); server.server_close()


def load_tests(loader, tests, pattern):
    # Reuse the HTTP harness without counting inherited baseline tests twice.
    suite = unittest.TestSuite()
    for name in SecurityHTTPTests.__dict__:
        if name.startswith('test_'): suite.addTest(SecurityHTTPTests(name))
    suite.addTests(loader.loadTestsFromTestCase(OutboundSecurityTests))
    return suite


if __name__ == '__main__': unittest.main()
