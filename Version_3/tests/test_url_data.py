import unittest

from phishing_gans_v3.url_data import LEXICAL_FEATURE_NAMES, canonicalize_url, lexical_features


class URLDataTests(unittest.TestCase):
    def test_http_and_https_canonicalize_identically(self) -> None:
        http = canonicalize_url("http://Example.COM/login?a=1")
        https = canonicalize_url("https://example.com/login?a=1")
        self.assertEqual(http[:2], https[:2])
        self.assertEqual(http[0], "example.com/login?a=1")
        self.assertEqual(http[2], "http")
        self.assertEqual(https[2], "https")

    def test_scheme_is_absent_from_features(self) -> None:
        self.assertNotIn("https", LEXICAL_FEATURE_NAMES)
        url = canonicalize_url("https://example.com/a")[0]
        self.assertEqual(len(lexical_features(url)), len(LEXICAL_FEATURE_NAMES))

    def test_user_information_is_removed(self) -> None:
        url, host, _ = canonicalize_url("https://user:password@example.com/path")
        self.assertEqual(url, "example.com/path")
        self.assertEqual(host, "example.com")

    def test_scheme_free_redirect_query_is_not_an_outer_scheme(self) -> None:
        value = "track.example.org/click?target=https://destination.example/login"
        url, host, scheme = canonicalize_url(value)
        self.assertEqual(url, value)
        self.assertEqual(host, "track.example.org")
        self.assertEqual(scheme, "http")


if __name__ == "__main__":
    unittest.main()
