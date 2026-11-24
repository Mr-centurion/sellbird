"""What a site's published addresses say about its email format, and a
pattern cache that no longer gets stuck on a default guess."""

import pytest

import sellbird.integrations.email_finder as ef


# ── email patterns from published addresses ──

@pytest.mark.parametrize("local, pattern", [
    ("j.smith", "{fi}.{l}"),
    ("john.smith", "{f}.{l}"),
    ("john_smith", "{f}_{l}"),
])
def test_pattern_from_an_untied_address(local, pattern):
    assert ef.pattern_from_local(local)[0] == pattern


@pytest.mark.parametrize("local", [
    "info", "office", "servicesubmissions", "scholarships", "web2", "x",
    # A lone dotless address is ambiguous: kmorrow@ could be Kmorrow or K. Morrow,
    # sven@ a first name or S. Ven. Nobody on the site to tell, nothing learned.
    "corwin", "wren", "jsmith", "kmorrow", "tvoss", "bpell", "rquade", "sven", "drsmith",
])
def test_role_or_junk_addresses_teach_nothing(local):
    assert ef.pattern_from_local(local) is None


def test_an_address_tied_to_a_named_person_beats_guessing():
    known, pattern, conf = ef.site_email_evidence(
        ["nvale@acme.com", "info@acme.com", "dana@other.com"],
        [("Nora", "Vale")], "acme.com")
    assert known == [("nvale@acme.com", "Nora", "Vale")]
    assert pattern == "{fi}{l}" and conf >= 0.9


def test_untied_dotless_addresses_teach_nothing():
    known, pattern, conf = ef.site_email_evidence(
        ["cora@lanternagency.test", "wren@lanternagency.test", "kmorrow@lanternagency.test"],
        [("Nate", "Calloway")], "lanternagency.test")
    assert (known, pattern) == ([], "")


def test_a_named_surname_disambiguates_an_untied_address():
    # Nobody named Jake Smith, but Amy Smith works there: jsmith@ is an
    # initial plus a surname, not a first name.
    known, pattern, conf = ef.site_email_evidence(
        ["jsmith@acme.com"], [("Amy", "Smith"), ("Nora", "Vale")], "acme.com")
    assert known == [] and pattern == "{fi}{l}" and conf < 0.9


@pytest.mark.parametrize("address, pattern", [("jsmith@acme.com", "{fi}{l}"), ("smith@acme.com", "{l}")])
def test_an_address_two_named_people_could_own_is_format_evidence_only(address, pattern):
    known, learned, _ = ef.site_email_evidence(
        [address], [("John", "Smith"), ("Jane", "Smith")], "acme.com")
    assert known == [] and learned == pattern


@pytest.mark.asyncio
async def test_never_guess_an_address_another_named_person_could_own(monkeypatch):
    async def mx(domain):
        return "aspmx.l.google.com"

    async def verified(email, mx_type, env):
        raise AssertionError("a colliding address must never be sent to a verifier")

    monkeypatch.setattr(ef, "get_mx_host", mx)
    monkeypatch.setattr(ef, "_verify_candidate", verified)
    st = PatternState({"pattern": "{fi}{l}", "source": "site_address", "confidence": 0.9})
    result = await ef.find_email("Jane", "Smith", "acme.com", state=st,
                                 other_people=[("John", "Smith")])
    assert result.email == "" and result.status != "verified"
    st = PatternState({"pattern": "{l}", "source": "site_address", "confidence": 0.9})
    result = await ef.find_email("Jane", "Smith", "acme.com", state=st,
                                 other_people=[("John", "Smith")])
    assert result.email == ""


@pytest.mark.asyncio
async def test_another_persons_published_address_is_never_this_persons_guess(monkeypatch):
    async def mx(domain):
        return "aspmx.l.google.com"

    monkeypatch.setattr(ef, "get_mx_host", mx)
    st = PatternState({"pattern": "{f}", "source": "site_address", "confidence": 0.9})
    result = await ef.find_email(
        "Nora", "Quist", "acme.com", verify=False, state=st, other_people=[("Nora", "Vale")],
        known_emails=[("nora@acme.com", "Nora", "Vale")])
    assert result.email == ""


# ── pattern derivation no longer gets stuck on a default ──

class PatternState:
    def __init__(self, cached=None):
        self.cached = cached
        self.saved = []

    async def get_email_pattern(self, domain):
        return self.cached

    async def save_email_pattern(self, domain, pattern="", source="", confidence=0.0, **kw):
        self.saved.append((pattern, source, confidence))

    async def get_setting(self, key, default=None):
        return "0"

    async def increment_setting(self, key):
        pass


@pytest.mark.asyncio
async def test_a_cached_default_does_not_block_a_known_address():
    st = PatternState({"pattern": "{f}.{l}", "source": "default", "confidence": 0.3})
    pattern, source, _ = await ef.derive_pattern(
        "acme.com", known_emails=[("nvale@acme.com", "Nora", "Vale")], state=st)
    assert (pattern, source) == ("{fi}{l}", "known_email")


@pytest.mark.asyncio
async def test_a_small_business_with_no_evidence_defaults_to_first_name(monkeypatch):
    pattern, source, conf = await ef.derive_pattern(
        "acme.com", state=PatternState(), default_pattern=ef.SMALL_BUSINESS_PATTERN)
    assert (pattern, source) == ("{f}", "default") and conf < 0.5


@pytest.mark.asyncio
async def test_a_persons_own_published_address_is_used_as_is(monkeypatch):
    async def mx(domain):
        return "aspmx.l.google.com"

    monkeypatch.setattr(ef, "get_mx_host", mx)
    result = await ef.find_email(
        "Tess", "Marlowe", "marlowemarketingco.test", verify=False, state=PatternState(),
        known_emails=[("tess@marlowemarketingco.test", "Tess", "Marlowe")])
    assert result.email == "tess@marlowemarketingco.test"
    assert result.status == "guess"   # unverified stays a guess: never sent
