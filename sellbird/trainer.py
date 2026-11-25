"""Trainer — give Sellbird a URL and it learns everything about the product.

Uses a local recursive crawler (httpx + BeautifulSoup) to discover and scrape pages.
"""

import asyncio
import copy
import logging
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from sellbird.brain import Brain
from sellbird.config import BrainConfig, load_brain_config
from sellbird.public_http import UnsafeURLError, public_http_client
from sellbird.state import StateManager

logger = logging.getLogger("sellbird.trainer")

MAX_OBJECTION_ATTEMPTS = 3

from sellbird.paths import PROJECT_ROOT
from sellbird.textio import read_user_text
from sellbird.untrusted import UNTRUSTED_RULE, fence


def _normalize_start_url(url: str) -> str:
    """Accept 'example.com' as well as 'https://example.com'."""
    url = url.strip()
    if url and not urlparse(url).scheme:
        url = f"https://{url}"
    return url

# Priority paths to try when crawling a website
FALLBACK_PATHS = [
    "/", "/about", "/about-us", "/pricing", "/features", "/product",
    "/products", "/solutions", "/services", "/how-it-works", "/why-us",
    "/case-studies", "/customers", "/testimonials", "/faq", "/contact",
    "/team", "/our-team", "/blog", "/resources", "/integrations",
]


class WebCrawler:
    """Recursive HTTP crawler using httpx and BeautifulSoup.

    Starts with priority pages, then follows every internal link it finds
    on each crawled page until it hits the max_pages limit.
    """

    # Set when a page was refused for pointing at a local or private address.
    blocked = False

    async def crawl(self, base_url: str, max_pages: int = 100) -> dict[str, str]:
        """Recursively crawl a website by following internal links."""
        self.blocked = False
        pages: dict[str, str] = {}
        visited: set[str] = set()
        domain = urlparse(base_url).netloc

        # Queue starts with priority paths, then discovered links
        queue: list[str] = [urljoin(base_url, p) for p in FALLBACK_PATHS]

        async with public_http_client(
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        ) as client:
            while queue and len(pages) < max_pages:
                url = queue.pop(0)

                # Normalize and skip if already visited
                clean_url = self._normalize_url(url)
                if clean_url in visited:
                    continue
                visited.add(clean_url)

                # Fetch the page
                html, text = await self._fetch_page(client, url)
                if not text:
                    continue

                pages[url] = text
                print(f"\r      Scraped {len(pages)} pages...", end="", flush=True)

                # Discover new links from this page and add to queue
                if html:
                    new_links = self._extract_links(html, base_url, domain)
                    for link in new_links:
                        norm = self._normalize_url(link)
                        if norm not in visited and link not in queue:
                            queue.append(link)

        print()  # newline after progress
        return pages

    async def _fetch_page(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[str, str]:
        """Fetch a page. Returns (raw_html, clean_text)."""
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return "", ""

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                return "", ""

            raw_html = resp.text
            soup = BeautifulSoup(raw_html, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "footer", "noscript", "iframe"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return raw_html, "\n".join(lines)

        except UnsafeURLError:
            self.blocked = True
            return "", ""
        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return "", ""

    def _extract_links(self, html: str, base_url: str, domain: str) -> list[str]:
        """Extract all internal links from an HTML page."""
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Skip anchors, mailto, tel, javascript
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue

            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # Only internal links
            if parsed.netloc == domain and parsed.scheme in ("http", "https"):
                # Strip query params and fragments for cleaner crawling
                clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                # Skip file downloads
                if not any(clean.endswith(ext) for ext in (".pdf", ".zip", ".png", ".jpg", ".svg", ".css", ".js")):
                    if clean not in links:
                        links.append(clean)

        return links

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for dedup (strip trailing slash, query, fragment)."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"


def brain_section(brain_config: BrainConfig) -> dict:
    """The ``brain:`` block to write back, so retraining keeps the choice."""
    section = {"provider": brain_config.provider}
    if brain_config.model:
        section["model"] = brain_config.model
    if brain_config.reasoning_effort:
        section["reasoning_effort"] = brain_config.reasoning_effort
    return section


# What training learns from a website. Retraining replaces exactly these and
# keeps everything else in the existing config (email provider, quiet hours,
# send limits, compliance, offer, the sender's name, email and tone, ...),
# which are the operator's choices.
LEARNED_FIELDS = {
    "persona": ("company",),
    "product": ("name", "description", "pricing", "key_benefits", "objection_responses"),
    "icp": ("industries", "company_size", "titles", "geography"),
}
# The shipped template's placeholder company: a persona still carrying it
# was never set up, so training fills in the whole persona.
UNTRAINED_COMPANY = "Your Company"


def _persona_untrained(persona) -> bool:
    return (
        not isinstance(persona, dict)
        or persona.get("company") == UNTRAINED_COMPANY
        or not persona.get("email")
    )


def merge_trained_config(learned: dict, existing: dict | None) -> dict:
    """``existing`` with what training learned laid over it.

    Sections the existing config lacks come from ``learned`` (its defaults).
    The brain selection always comes from the trainer, which already starts
    from the existing choice unless ``--provider`` overrode it.
    """
    if not isinstance(existing, dict) or not existing:
        return learned
    merged = copy.deepcopy(existing)
    untrained = _persona_untrained(existing.get("persona"))
    for section, value in learned.items():
        if not isinstance(merged.get(section), dict):
            merged[section] = copy.deepcopy(value)
    for section, keys in LEARNED_FIELDS.items():
        for key in keys:
            if key in learned.get(section, {}):
                merged[section][key] = copy.deepcopy(learned[section][key])
    if untrained:
        merged["persona"].update(copy.deepcopy(learned.get("persona", {})))
    # The offer is yours: retraining refreshes the ideas and never the offer.
    learned_offer = (learned.get("product") or {}).get("offer") or {}
    if learned_offer.get("entry_ideas") and isinstance(merged.get("product"), dict):
        offer = merged["product"].setdefault("offer", {})
        if isinstance(offer, dict):
            offer["entry_ideas"] = list(learned_offer["entry_ideas"])
    if "brain" in learned:
        merged["brain"] = copy.deepcopy(learned["brain"])
    # The trained site joins your own domains (autopilot may link to it);
    # the ones already listed stay.
    trained = ((learned.get("channels") or {}).get("email") or {}).get("link_domains") or []
    if trained and isinstance(merged.get("channels"), dict):
        email = merged["channels"].setdefault("email", {})
        if isinstance(email, dict):
            have = list(email.get("link_domains") or [])
            email["link_domains"] = have + [d for d in trained if d not in have]
    return merged


def _offer_from(product_info: dict) -> dict:
    """The offer ideas training found. Training never picks the offer: the
    user works it out with their agent and confirms it
    on Your company."""
    ideas = product_info.get("entry_offer_ideas")
    ideas = [str(i).strip() for i in ideas if str(i).strip()][:4] if isinstance(ideas, list) else []
    return {"entry_ideas": ideas}


def _backup(path: Path, raw: str) -> Path:
    """Save ``raw`` next to ``path`` without overwriting an earlier backup."""
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.{stamp}.bak")
    backup.write_text(raw, encoding="utf-8")
    return backup


def write_trained_config(learned: dict, path: Path) -> dict:
    """Write a trained config to ``path`` without losing the operator's settings.

    The base is ``path`` itself when it exists. On a first training it is the
    sellbird.yaml next to it (the template Sellbird ships, with the provider and
    compliance blocks), never a config found elsewhere on disk. A base that
    has content but can't be read as YAML is backed up before it is
    replaced. The new file is swapped in whole, so a failed write never
    leaves a half-written config. Returns what was written.
    """
    path = Path(path)
    source = path if path.exists() else path.with_name("sellbird.yaml")
    existing = None
    if source.exists():
        raw = read_user_text(source)
        try:
            existing = yaml.safe_load(raw)
        except yaml.YAMLError:
            existing = None
        if not isinstance(existing, dict) and source == path and raw.strip():
            backup = _backup(path, raw)
            print(f"      Couldn't read the existing {path.name}; saved it as {backup.name}.")
    config = merge_trained_config(learned, existing)
    if config is not learned and source == path:
        print(f"      Kept your settings from {path.name} (email provider, schedule, limits,")
        print("      compliance, sender identity); updated the product, targeting and company.")

    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        if path.exists():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return config


FallbackCrawler = WebCrawler


class Trainer:
    def __init__(
        self,
        brain_config: BrainConfig | None = None,
        state: StateManager | None = None,
    ):
        # Without an explicit choice, train with whatever provider the
        # existing config selects (Ollama when none does).
        self.brain_config = brain_config or load_brain_config()
        self._state = state or StateManager()
        self.brain = Brain(self._state, self.brain_config)
        self.source_name: str = ""
        self.source_folder: str = ""
        self.scraped_pages: dict[str, str] = {}

    @property
    def state(self) -> StateManager:
        return self._state

    @state.setter
    def state(self, value: StateManager) -> None:
        self._state = value
        if hasattr(self, "brain") and self.brain is not None:
            self.brain.state = value

    async def train(
        self,
        url: str,
        output_path: str = "sellbird.local.yaml",
        max_pages: int = 100,
    ):
        """Train Sellbird on a product website.

        1. Deep crawl the site (local crawler)
        2. Analyze all content with the configured brain
        3. Extract product info, ICP, objections, competitive intel
        4. Generate complete config + product knowledge skill
        """
        load_dotenv()
        await self.state.init_db()

        url = _normalize_start_url(url)
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain:
            print(f"\nERROR: '{url}' doesn't look like a valid URL.")
            print("Try something like: https://yourcompany.com")
            return None
        self.source_name = domain.split(".")[0].replace("-", " ").replace("_", " ").title()
        base_url = f"{parsed.scheme}://{domain}"

        logger.info(f"Training Sellbird on {base_url}...")
        print(f"\n{'='*60}")
        print(f"  Sellbird Training Mode")
        print(f"  Learning from: {base_url}")
        print(f"{'='*60}\n")

        # Step 1: Crawl website
        print(f"[1/6] Crawling website (up to {max_pages} pages)...")
        crawler = WebCrawler()
        self.scraped_pages = await crawler.crawl(base_url, max_pages=max_pages)

        if not self.scraped_pages:
            if getattr(crawler, "blocked", False):
                print(f"\nERROR: {base_url} points to a local or private address. Sellbird only")
                print("learns from public websites, so it never reaches into your own network.")
                print("Train on the live public site instead.")
            else:
                print("\nERROR: Could not scrape any pages. Check the URL.")
            return None

        print(f"      Done! Crawled {len(self.scraped_pages)} pages.\n")
        return await self._run_analysis(domain=domain, output_path=output_path)

    async def train_from_folder(
        self,
        folder_path: str,
        output_path: str = "sellbird.local.yaml",
    ):
        """Train Sellbird from local .md/.txt files in a folder instead of crawling."""
        load_dotenv()
        await self.state.init_db()

        p = Path(folder_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p

        if not p.exists() or not p.is_dir():
            print(f"\nERROR: '{folder_path}' is not a valid directory.")
            return None

        self.source_folder = str(p)
        self.source_name = p.name.replace("sample_", "").replace("-", " ").replace("_", " ").title()

        print(f"\n{'='*60}")
        print(f"  Sellbird Training Mode (Folder Loader)")
        print(f"  Reading from: {p}")
        print(f"{'='*60}\n")

        files = sorted([f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in (".md", ".txt")])
        if not files:
            print(f"\nERROR: No .md or .txt files found in {folder_path}.")
            return None

        self.scraped_pages = {}
        for f in files:
            rel = str(f.relative_to(p))
            try:
                self.scraped_pages[rel] = f.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read {f}: {e}")

        if not self.scraped_pages:
            print(f"\nERROR: Could not read any files in {folder_path}.")
            return None

        print(f"[1/6] Loaded {len(self.scraped_pages)} local files from {p.name}...\n")
        domain = f"{p.name.lower().replace('_', '-')}.local"
        return await self._run_analysis(domain=domain, output_path=output_path)

    async def generate_icp_for_product(self, product: dict) -> dict:
        """Generate an ICP for a specific product."""
        return await self._extract_icp(product=product)

    async def _run_analysis(
        self,
        domain: str,
        output_path: str = "sellbird.local.yaml",
    ) -> dict | None:
        """Run the training analysis pipeline over self.scraped_pages."""
        # Step 2: Extract product information
        print("[2/6] Analyzing product information...")
        product_info = await self._extract_product_info()
        if not product_info:
            print("ERROR: Could not extract product information.")
            return None
        print(f"      Found: {product_info.get('product_name', 'Unknown Product')}\n")

        # Multi-product persistence: save individual products and per-product ICPs
        model_name = getattr(getattr(self.brain, "config", None), "model", "local")
        products_list = product_info.get("products") or []
        if products_list:
            for prod in products_list:
                p_name = prod.get("name") or "Product"
                p_desc = prod.get("description") or ""
                p_val = prod.get("value_props") or prod.get("value_proposition") or ""
                p_id = re.sub(r"[^a-z0-9_-]", "_", p_name.lower()).strip("_") or f"prod_{uuid.uuid4().hex[:8]}"
                await self.state.add_product(
                    name=p_name,
                    description=p_desc,
                    value_props=p_val,
                    product_id=p_id,
                )
                prod_icp = await self._extract_icp(product=prod)
                await self.state.save_icp(product_id=p_id, icp_data=prod_icp, model=model_name)

        # Step 3: Identify primary ICP
        print("[3/6] Identifying ideal customer profile...")
        icp_info = await self._extract_icp()
        print(f"      Target: {', '.join(icp_info.get('industries', ['Unknown']))}\n")

        # Save default product / primary ICP for backward compatibility
        await self.state.save_icp(product_id="default", icp_data=icp_info, model=model_name)
        if not products_list:
            await self.state.add_product(
                name=product_info.get("product_name", "Default Product"),
                description=product_info.get("product_description", ""),
                value_props=product_info.get("value_proposition", ""),
                product_id="default",
            )

        # Step 4: Competitive analysis
        print("[4/6] Analyzing competitive landscape...")
        competitive_intel = await self._extract_competitive_intel(product_info)
        print(f"      Identified {len(competitive_intel.get('competitors', []))} competitors.\n")

        # Step 5: Generate objection responses
        print("[5/6] Generating objection handling playbook...")
        objections = await self._generate_objections(
            product_info,
            competitive_intel,
        )

        if not objections:
            print("\nERROR: Could not generate a valid objection playbook.")
            print("Training stopped. Existing configuration was not modified.")
            return None

        print(f"      Prepared {len(objections)} objection responses.\n")

        # Step 6: Generate config + skills
        print("[6/6] Building configuration and product knowledge...")
        config = self._build_config(product_info, icp_info, objections, domain)

        # Write config (relative paths resolve against the project root,
        # so training works no matter what directory it's launched from)
        config_path = Path(output_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        guessed_email = config["persona"]["email"]
        config = write_trained_config(config, config_path)

        # Generate product knowledge skill
        await self._generate_product_knowledge(product_info, icp_info, competitive_intel)

        # Generate competitive battle cards skill
        if competitive_intel.get("competitors"):
            await self._generate_battle_cards(product_info, competitive_intel)

        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"{'='*60}")
        print(f"\n  Product:      {config['product']['name']}")
        print(f"  Company:      {config['persona']['company']}")
        print(f"  ICP titles:   {', '.join(config['icp']['titles'])}")
        print(f"  Industries:   {', '.join(config['icp']['industries'])}")
        print(f"  Pages crawled: {len(self.scraped_pages)}")
        offer = (config.get("product") or {}).get("offer") or {}
        if offer.get("entry"):
            print(f"\n  First-email offer: {offer['entry']}")
        else:
            print("\n  Next: work out the offer every first email ends with. Ask your coding agent")
            print("  to work it out with you, or open the dashboard's Your company tab.")
            if offer.get("entry_ideas"):
                print("  Ideas to start from: " + "; ".join(offer["entry_ideas"]))
        print(f"\n  Files generated:")
        print(f"    - {config_path}")
        print(f"    - skills/product_knowledge.md")
        if competitive_intel.get("competitors"):
            print(f"    - skills/competitive_intel.md")
        if config["persona"].get("email") == guessed_email:
            print(f"\n  Review {config_path} and adjust as needed — especially the")
            print(f"  persona name/email, which I guessed from your domain.")
        else:
            print(f"\n  Review {config_path} and adjust as needed. Emails still go out as")
            print(f"  {config['persona'].get('name', '')} <{config['persona']['email']}>.")
        print(f"  Then run: sellbird run\n")

        return config

    def _get_all_content(self, max_chars: int = 60000) -> str:
        """All scraped pages as one <untrusted> block, for analysis.

        Even your own site can carry text you didn't write (comments,
        reviews, a page someone else controls), and what training learns
        goes into every email afterwards, so the pages are fenced as data."""
        all_content = ""
        for url, content in self.scraped_pages.items():
            path = urlparse(url).path or "/"
            # Truncate individual pages to keep total manageable
            page_content = content[:8000] if len(content) > 8000 else content
            all_content += f"\n\n--- PAGE: {path} ---\n{page_content}"
            if len(all_content) > max_chars:
                break
        return fence(all_content, "your website")

    def _extract_company_name_deterministic(self) -> str:
        """Extract company name from folder, domain, or stored text headers."""
        if getattr(self, "source_name", None):
            return self.source_name

        # Check scraped pages keys (domains or folder paths)
        for key in self.scraped_pages.keys():
            if "://" in key:
                parsed = urlparse(key)
                host = parsed.hostname or ""
                host = re.sub(r"^www\.", "", host)
                name = host.split(".")[0]
                if name and name not in ("localhost", "127"):
                    return name.replace("-", " ").replace("_", " ").title()
            elif "/" in key or "\\" in key or key.endswith((".txt", ".md", ".html")):
                p = Path(key)
                if p.parent and p.parent.name and p.parent.name not in (".", "companies"):
                    name = p.parent.name.replace("sample_", "").replace("-", " ").replace("_", " ").title()
                    if name:
                        return name

        # Look in stored text for Company Overview or About headings
        all_raw = " ".join(self.scraped_pages.values())
        m = re.search(r"(?:SAMPLE DATA:\s*)?(?:About\s+)([A-Za-z0-9&.,' -]{2,50}?)(?:\n|\.|$)", all_raw)
        if m:
            candidate = m.group(1).strip().rstrip(".,")
            if candidate and not any(w in candidate.lower() for w in ("our product", "our pricing", "overview", "us", "website")):
                return candidate

        m2 = re.search(r"(?:About|Company(?:\s+Overview)?):\s*([A-Za-z0-9&.,' -]{2,50}?)(?:\n|\.|$)", all_raw)
        if m2:
            candidate = m2.group(1).strip().rstrip(".,")
            if candidate and not any(w in candidate.lower() for w in ("our product", "our pricing", "overview", "us", "website")):
                return candidate

        return "Business"

    def _extract_products_deterministic(self, all_content: str, company_name: str) -> list[dict]:
        """Extract products from markdown headings or 'Product:' lines in stored text."""
        products = []
        seen_names = set()

        # Pattern 1: Lines like "Product 1: Name" or "Product: Name"
        matches = list(re.finditer(r"(?im)^Product(?:\s+\d+)?:?\s*([A-Za-z0-9][A-Za-z0-9 &.,'/-]{1,60})$", all_content))
        if matches:
            for i, m in enumerate(matches):
                clean = m.group(1).strip().rstrip(":-")
                clean_lower = clean.lower()
                if clean and clean_lower not in seen_names and not any(w in clean_lower for w in ("catalog", "overview", "specification", "feature", "pricing")):
                    seen_names.add(clean_lower)
                    start_pos = m.start()
                    end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(all_content)
                    block = all_content[start_pos:end_pos]

                    desc_m = re.search(r"(?:Description|Overview):\s*([^\n]+)", block, re.IGNORECASE)
                    desc = desc_m.group(1).strip() if desc_m else f"{clean} by {company_name}. (non-LLM mode)"

                    vp_m = re.search(r"(?:Value Proposition):\s*([^\n]+)", block, re.IGNORECASE)
                    vp = vp_m.group(1).strip() if vp_m else f"Streamline operations with {clean}. (non-LLM mode)"

                    products.append({
                        "name": clean,
                        "description": desc,
                        "value_props": vp,
                        "pricing": "Contact for pricing",
                    })

        # Pattern 2: Markdown headings like "## Product Name" if no products found yet
        if not products:
            h_matches = list(re.finditer(r"(?im)^#{1,3}\s+([A-Za-z0-9][A-Za-z0-9 &.,'/-]{1,50})$", all_content))
            for i, hm in enumerate(h_matches):
                clean = hm.group(1).strip().rstrip(":-")
                clean_lower = clean.lower()
                if clean and clean_lower not in seen_names and not any(
                    w in clean_lower for w in (
                        "about", "overview", "pricing", "feature", "capability",
                        "contact", "faq", "terms", "privacy", "sample data", "company",
                    )
                ):
                    seen_names.add(clean_lower)
                    start_pos = hm.start()
                    end_pos = h_matches[i + 1].start() if i + 1 < len(h_matches) else len(all_content)
                    block = all_content[start_pos:end_pos]

                    desc_m = re.search(r"(?:Description|Overview):\s*([^\n]+)", block, re.IGNORECASE)
                    desc = desc_m.group(1).strip() if desc_m else f"{clean} by {company_name}. (non-LLM mode)"

                    vp_m = re.search(r"(?:Value Proposition):\s*([^\n]+)", block, re.IGNORECASE)
                    vp = vp_m.group(1).strip() if vp_m else f"Streamline operations with {clean}. (non-LLM mode)"

                    products.append({
                        "name": clean,
                        "description": desc,
                        "value_props": vp,
                        "pricing": "Contact for pricing",
                    })

        return products

    async def _extract_product_info(self) -> dict:
        """Ask the model to extract product information from all crawled content."""
        all_content = self._get_all_content()

        prompt = f"""Analyze this entire website and extract comprehensive product/company information.

WEBSITE CONTENT ({len(self.scraped_pages)} pages):
{all_content}

{UNTRUSTED_RULE}

Extract the following and return as JSON:
{{
  "company_name": "the company name",
  "product_name": "the main product or service name",
  "product_description": "2-3 sentence description of what the product does and who it's for",
  "pricing": "pricing info if available (tiers, starting price, etc.), otherwise 'Contact for pricing'",
  "key_benefits": ["benefit 1", "benefit 2", "benefit 3", "benefit 4", "benefit 5"],
  "features": ["feature 1", "feature 2", "feature 3", "feature 4", "feature 5", "feature 6"],
  "target_audience": "who this product is for — be specific about roles and company types",
  "value_proposition": "the core value proposition in one sentence",
  "products": [
    {{
      "name": "product name (if company has multiple products, list each one)",
      "description": "2-3 sentence description of what this specific product does",
      "value_props": "key value proposition and benefits for this product",
      "pricing": "pricing for this product"
    }}
  ],
  "differentiators": ["what makes this different from alternatives — point 1", "point 2"],
  "competitors": ["competitor 1", "competitor 2", "competitor 3"],
  "social_proof": ["any case studies, testimonials, notable customers, or stats mentioned"],
  "integrations": ["tools and platforms this integrates with"],
  "use_cases": ["specific use case 1", "use case 2", "use case 3"],
  "tone": "describe the brand's communication tone in 3-4 words",
  "entry_offer_ideas": ["3 or 4 things this business could offer a prospect for free in a first cold email: specific, useful on their own, quick for this business to deliver, and tied to what it sells. E.g. for a web designer 'a mockup of your homepage with online booking', for an SEO agency 'a one-page report on where you rank and why', for inventory software 'a reorder forecast for your top 20 SKUs'. Never a discount or a generic call."]
}}

Be thorough — you have access to the full website. Extract everything relevant."""

        result = await self.brain.think_json(
            prompt, session_id="sellbird-trainer", agent="trainer", task="extract_product_info"
        )
        if not isinstance(result, dict) or not result:
            raw_text = "".join(self.scraped_pages.values()).strip()
            if not raw_text:
                raise ValueError("Could not extract product information: stored text is empty.")
            all_txt = self._get_all_content()

            company_name = self._extract_company_name_deterministic()
            products_list = self._extract_products_deterministic(all_txt, company_name)

            if not products_list:
                primary_name = f"{company_name} Platform"
                products_list = [{
                    "name": primary_name,
                    "description": f"{company_name} core operations and business services. (non-LLM mode)",
                    "value_props": f"Reliable operational solutions by {company_name}. (non-LLM mode)",
                    "pricing": "Contact for pricing",
                }]

            primary_prod = products_list[0]
            return {
                "company_name": company_name,
                "product_name": primary_prod["name"],
                "product_description": primary_prod["description"],
                "pricing": primary_prod.get("pricing", "Contact for pricing"),
                "key_benefits": [
                    f"Operational efficiency with {primary_prod['name']}",
                    "Reduced manual administrative overhead",
                    "Seamless cross-team coordination",
                ],
                "features": [
                    f"{primary_prod['name']} core engine",
                    "Automated error handling and monitoring",
                    "Data sync and status reporting",
                ],
                "target_audience": f"Operations and business leaders seeking {primary_prod['name']} capabilities",
                "value_proposition": primary_prod["value_props"],
                "products": products_list,
                "mode": "non-LLM mode",
            }
        return result

    async def _extract_icp(self, product: dict | None = None) -> dict:
        """Ask the model to determine the ideal customer profile from all content or a specific product."""
        all_content = self._get_all_content(max_chars=40000)

        product_clause = ""
        if product:
            p_name = product.get("name") or product.get("product_name", "")
            p_desc = product.get("description") or product.get("product_description", "")
            p_val = product.get("value_props") or product.get("value_proposition", "")
            product_clause = f"\nFocus on the Ideal Customer Profile specifically for this product:\nProduct Name: {p_name}\nDescription: {p_desc}\nValue Proposition: {p_val}\n"

        prompt = f"""Based on this content, determine the Ideal Customer Profile (ICP).
{product_clause}
Look at case studies, testimonials, pricing tiers, feature descriptions, and target messaging
to understand exactly who this product is built for.

WEBSITE CONTENT ({len(self.scraped_pages)} pages):
{all_content}

{UNTRUSTED_RULE}

Return as JSON:
{{
  "industries": ["industry 1", "industry 2", "industry 3"],
  "company_size": "employee range like '10-200 employees'",
  "titles": ["decision maker title 1", "title 2", "title 3", "title 4"],
  "geography": ["country or region 1", "region 2"],
  "buyer_persona": "A detailed description of the ideal buyer — their role, daily challenges, goals, and what success looks like for them",
  "pain_points": ["specific pain point 1", "pain point 2", "pain point 3", "pain point 4"],
  "buying_triggers": ["trigger event that makes them ready to buy 1", "trigger 2", "trigger 3"],
  "disqualifiers": ["signals that someone is NOT a good fit 1", "disqualifier 2"]
}}

For any city in "geography", always include the state/province and country if
the name could plausibly refer to more than one place — write "Solon, Ohio"
not "Solon" (there's also a Solon in France), "Springfield, Illinois" not
"Springfield" (30+ US cities share that name). These strings get geocoded
automatically for local-business discovery, and an unqualified name silently
resolves to the wrong place on Earth instead of failing loudly.

"titles" MUST be literal job titles exactly as they'd appear on a business
card or LinkedIn profile ("Owner", "CEO", "President", "Plant Manager",
"Practice Manager") — never a customer-segment or persona label ("Business
Owner Segment", "Decision Maker", "Healthcare Professional"). Sellbird matches
a found contact's title against this list by substring, so a descriptive
label instead of a real title means every real contact silently fails to
match and gets discarded. If the buyer is an INDIVIDUAL bought into by
persona rather than a role at a target company (e.g. this product targets
physicians, business owners, or executives directly as people, not a
"Director of X" at a company you sell software to), list every real job
title that persona plausibly holds — a "business owner" persona should
produce ["Owner", "President", "CEO", "Founder", "Managing Partner"], a
"physician" persona should produce ["Physician", "MD", "Doctor"], and so on.
Prefer more titles over fewer; a missed real title costs a contact forever,
an extra one costs nothing.

Be specific — don't guess generically. Base this on what the website actually says about their customers."""

        result = await self.brain.think_json(
            prompt, session_id="sellbird-trainer", agent="trainer", task="extract_icp"
        )
        if not isinstance(result, dict) or not result:
            raw_text = "".join(self.scraped_pages.values()).strip()
            if not raw_text:
                raise ValueError("Could not extract ICP: stored text is empty.")
            all_content = self._get_all_content()

            p_name = ""
            p_desc = ""
            if product:
                p_name = (product.get("name") or product.get("product_name") or "").strip()
                p_desc = (product.get("description") or product.get("product_description") or "").strip()

            target_text = f"{all_content}\n{p_name}\n{p_desc}".lower()

            # 1. Industries from keywords in target text
            industry_kw_map = [
                (r"\b(software|saas|developer|code|api|cloud|devops|telemetry|dora)\b", "Software"),
                (r"\b(logistics|freight|warehouse|fulfillment|shipping|supply chain)\b", "Logistics"),
                (r"\b(manufacturing|factory|plant|industrial)\b", "Manufacturing"),
                (r"\b(healthcare|clinical|medical|hospital|patient|health)\b", "Healthcare"),
                (r"\b(fintech|banking|financial|payments|investment)\b", "Financial Services"),
                (r"\b(e-?commerce|retail|store|merchandise)\b", "E-commerce"),
                (r"\b(cybersecurity|security|infosec|compliance)\b", "Cybersecurity"),
                (r"\b(education|edtech|learning|school|university)\b", "Education"),
                (r"\b(telecom|telecommunications|network)\b", "Telecommunications"),
                (r"\b(construction|contractor|architecture)\b", "Construction"),
                (r"\b(marketing|advertising|media)\b", "Marketing"),
            ]
            industries = []
            for pattern, ind_name in industry_kw_map:
                if re.search(pattern, target_text):
                    if ind_name not in industries:
                        industries.append(ind_name)

            if not industries:
                industries = ["Technology", "Business Services"]

            # 2. Decision Maker Titles from functional keywords
            titles = []
            if any(w in target_text for w in ("engineering", "developer", "dora", "telemetry", "devops", "code", "software")):
                titles.extend(["VP Engineering", "CTO", "Director of Engineering", "Head of DevOps"])
            if any(w in target_text for w in ("logistics", "supply chain", "warehouse", "fulfillment", "operations", "order", "erp")):
                titles.extend(["VP Operations", "Director of Supply Chain", "Head of Logistics", "COO"])
            if any(w in target_text for w in ("security", "ciso", "compliance", "threat")):
                titles.extend(["CISO", "VP Security", "Director of Information Security"])
            if any(w in target_text for w in ("clinical", "medical", "physician", "patient")):
                titles.extend(["Chief Medical Officer", "Director of Clinical Operations", "Practice Manager"])
            if any(w in target_text for w in ("financial", "cfo", "accounting", "treasury")):
                titles.extend(["CFO", "VP Finance", "Director of Finance"])

            if not titles:
                titles = ["VP Operations", "Director", "Head of Department", "Founder", "CEO"]

            seen_titles = set()
            dedup_titles = []
            for t in titles:
                if t not in seen_titles:
                    seen_titles.add(t)
                    dedup_titles.append(t)
            titles = dedup_titles[:6]

            # 3. Company Size
            size_match = re.search(r"\b(\d+\s*[-–]\s*\d+\s*(?:employees|engineers|staff|people)?)\b", target_text)
            if size_match:
                company_size = size_match.group(1).strip()
                if "employee" not in company_size and "engineer" not in company_size:
                    company_size += " employees"
            elif any(w in target_text for w in ("enterprise", "fortune", "global")):
                company_size = "250-2,500+ employees"
            elif any(w in target_text for w in ("mid-market", "growth", "scale")):
                company_size = "50-500 employees"
            else:
                company_size = "10-200 employees"

            # 4. Pain Points from text lines or keywords
            pain_keywords = ["bottleneck", "delay", "friction", "manual", "error", "inefficiency", "overhead", "flaky", "discrepanc", "sync"]
            pain_points = []
            for line in all_content.splitlines():
                line_clean = line.strip().lstrip("-*#0123456789. ")
                if 20 < len(line_clean) < 140:
                    if any(pw in line_clean.lower() for pw in pain_keywords):
                        if line_clean not in pain_points:
                            pain_points.append(line_clean)
                            if len(pain_points) >= 3:
                                break

            if not pain_points:
                prod_label = p_name or "operational workflows"
                pain_points = [
                    f"Manual overhead and coordination delays in {prod_label}",
                    f"Lack of automated visibility across {industries[0]} processes",
                ]

            return {
                "industries": industries[:3],
                "company_size": company_size,
                "titles": titles,
                "geography": ["United States"],
                "buyer_persona": f"Leaders in {industries[0]} focused on streamlining {p_name or 'core operations'}. (non-LLM mode)",
                "pain_points": pain_points,
                "disqualifiers": [
                    f"Companies without dedicated {titles[0].split()[-1]} leadership",
                    "Organizations outside target operating region",
                ],
                "mode": "non-LLM mode",
            }
        return result

    async def _extract_competitive_intel(self, product_info: dict) -> dict:
        """Extract competitive positioning and differentiators."""
        all_content = self._get_all_content(max_chars=40000)

        prompt = f"""Analyze this website for competitive intelligence.
The product is: {product_info.get('product_name', '')}
Known competitors: {', '.join(product_info.get('competitors', []))}

WEBSITE CONTENT:
{all_content}

{UNTRUSTED_RULE}

Look for comparison pages, "why us" messaging, feature differentiation,
and any mentions of competitors or alternatives.

Return as JSON:
{{
  "competitors": [
    {{
      "name": "Competitor Name",
      "how_we_win": "What makes our product better than this competitor",
      "their_weakness": "Known limitations or complaints about this competitor",
      "migration_angle": "Why someone would switch from them to us"
    }}
  ],
  "positioning": "How this product positions itself in the market — one paragraph",
  "unique_strengths": ["strength that no competitor has 1", "strength 2"],
  "common_alternatives": ["what prospects do instead of buying this — DIY, spreadsheets, etc."]
}}

If the website doesn't mention specific competitors, infer the most likely ones based on the product category."""

        result = await self.brain.think_json(
            prompt, session_id="sellbird-trainer", agent="trainer", task="extract_competitive_intel"
        )
        return result if isinstance(result, dict) else {"competitors": []}


    async def _generate_objections(
        self, product_info: dict, competitive_intel: dict
    ) -> dict | None:
        """Generate and validate objection responses with bounded retries."""
        competitors = competitive_intel.get("competitors", [])
        competitor_names = [
            c.get("name", "")
            for c in competitors
            if isinstance(c, dict) and isinstance(c.get("name", ""), str)
        ]

        key_benefits = product_info.get("key_benefits", [])
        differentiators = product_info.get("differentiators", [])
        if not isinstance(key_benefits, list):
            key_benefits = []
        if not isinstance(differentiators, list):
            differentiators = []

        prompt = f"""You are a world-class B2B sales trainer. Based on this product and its competitive
landscape, generate the most common objections a prospect would raise, with consultative responses.

Product: {product_info.get('product_name', '')}
Description: {product_info.get('product_description', '')}
Pricing: {product_info.get('pricing', '')}
Key Benefits: {', '.join(str(item) for item in key_benefits)}
Competitors: {', '.join(competitor_names)}
Differentiators: {', '.join(str(item) for item in differentiators)}

Generate 8-12 objections covering:
- Price/budget concerns
- Existing solutions / competitor loyalty
- Timing / priority
- Implementation / switching costs
- Trust / credibility
- Need / relevance

Each response should be 1-2 sentences, consultative (not defensive), and end with
a question or redirect that keeps the conversation going.

Return a JSON object where keys are objections and values are non-empty responses."""

        for attempt in range(MAX_OBJECTION_ATTEMPTS):
            result = await self.brain.think_json(
                prompt,
                session_id="sellbird-trainer",
                agent="trainer",
                task="generate_objections",
            )

            # A provider outage is different from malformed model output.
            if getattr(self.brain, "last_reply_empty", False):
                logger.warning(
                    "Objection generation stopped: provider returned no response."
                )
                return None

            # Accept only a non-empty dictionary of non-empty string pairs.
            if isinstance(result, dict) and result:
                valid = all(
                    isinstance(objection, str)
                    and objection.strip()
                    and isinstance(response, str)
                    and response.strip()
                    for objection, response in result.items()
                )
                if valid:
                    cleaned = {
                        objection.strip(): response.strip()
                        for objection, response in result.items()
                    }
                    logger.info(
                        "Generated %s valid objection responses.", len(cleaned)
                    )
                    return cleaned

            if attempt < MAX_OBJECTION_ATTEMPTS - 1:
                logger.warning(
                    "Invalid objection playbook. Retrying attempt %s/%s.",
                    attempt + 1,
                    MAX_OBJECTION_ATTEMPTS,
                )
                prompt += """

IMPORTANT: Your previous response was invalid or empty.
Return a NON-EMPTY JSON object containing objection strings as keys
and non-empty consultative response strings as values.
Do not return an empty object, markdown, or explanations.
"""

        logger.error(
            "Objection generation failed after %s attempts.",
            MAX_OBJECTION_ATTEMPTS,
        )
        return None

    async def _generate_product_knowledge(
        self, product_info: dict, icp_info: dict, competitive_intel: dict
    ):
        """Generate comprehensive product knowledge skill file."""
        product_name = product_info.get("product_name", "the product")

        knowledge = f"""# Product Knowledge: {product_name}

## What We Sell
{product_info.get('product_description', '')}

## Core Value Proposition
{product_info.get('value_proposition', '')}

## Key Benefits
{chr(10).join('- ' + b for b in product_info.get('key_benefits', []))}

## Features
{chr(10).join('- ' + f for f in product_info.get('features', []))}

## Use Cases
{chr(10).join('- ' + u for u in product_info.get('use_cases', []))}

## Pricing
{product_info.get('pricing', 'Contact for pricing')}

## Target Audience
{product_info.get('target_audience', '')}

## Ideal Buyer Persona
{icp_info.get('buyer_persona', '')}

## Integrations
{chr(10).join('- ' + i for i in product_info.get('integrations', []))}

## What Makes Us Different
{chr(10).join('- ' + d for d in product_info.get('differentiators', []))}

## Social Proof & Case Studies
{chr(10).join('- ' + s for s in product_info.get('social_proof', []))}

## Prospect Pain Points
{chr(10).join('- ' + p for p in icp_info.get('pain_points', []))}

## Buying Triggers
When these events happen, prospects are most likely to buy:
{chr(10).join('- ' + t for t in icp_info.get('buying_triggers', []))}

## Disqualifiers
Do NOT pursue prospects who:
{chr(10).join('- ' + d for d in icp_info.get('disqualifiers', []))}

## Market Positioning
{competitive_intel.get('positioning', '')}

## Tone & Voice
{product_info.get('tone', 'professional, consultative, confident')}
"""

        skills_path = PROJECT_ROOT / "skills" / "product_knowledge.md"
        skills_path.write_text(knowledge, encoding="utf-8")
        print(f"      Generated: skills/product_knowledge.md")

    async def _generate_battle_cards(self, product_info: dict, competitive_intel: dict):
        """Generate competitive battle cards as a skill file."""
        competitors = competitive_intel.get("competitors", [])
        if not competitors:
            return

        product_name = product_info.get("product_name", "Our Product")

        cards = f"""# Competitive Battle Cards: {product_name}

Use these when a prospect mentions a competitor or asks "how are you different?"

## Market Positioning
{competitive_intel.get('positioning', '')}

## Our Unique Strengths
{chr(10).join('- ' + s for s in competitive_intel.get('unique_strengths', []))}

## Common Alternatives (Non-Competitors)
What prospects do instead of buying a solution like ours:
{chr(10).join('- ' + a for a in competitive_intel.get('common_alternatives', []))}

---
"""

        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            name = comp.get("name", "Unknown")
            cards += f"""
## vs. {name}

**How we win:** {comp.get('how_we_win', 'N/A')}

**Their weakness:** {comp.get('their_weakness', 'N/A')}

**Migration angle:** {comp.get('migration_angle', 'N/A')}

**When a prospect says "We use {name}":**
"That's a solid tool. The teams that've moved to {product_name} from {name} tell us the biggest difference is {comp.get('how_we_win', 'the experience')}. Worth a quick comparison?"

---
"""

        skills_path = PROJECT_ROOT / "skills" / "competitive_intel.md"
        skills_path.write_text(cards, encoding="utf-8")
        print(f"      Generated: skills/competitive_intel.md")

    def _build_config(
        self,
        product_info: dict,
        icp_info: dict,
        objections: dict,
        domain: str,
    ) -> dict:
        """Build the complete sellbird.yaml config."""
        company = product_info.get("company_name", "Your Company")
        product = product_info.get("product_name", "Your Product")
        tone = product_info.get("tone", "professional, consultative, confident")
        # "www.example.com" should yield alex@example.com, not alex@www.example.com
        email_domain = domain.removeprefix("www.")

        return {
            "persona": {
                "name": "Alex",
                "company": company,
                "role": "Business Development",
                "email": f"alex@{email_domain}",
                "linkedin": "",
                "tone": tone,
            },
            "product": {
                "name": product,
                "description": product_info.get("product_description", ""),
                "pricing": product_info.get("pricing", "Contact for pricing"),
                "key_benefits": product_info.get("key_benefits", []),
                "objection_responses": objections,
                "offer": _offer_from(product_info),
            },
            "icp": {
                "industries": icp_info.get("industries", ["Technology"]),
                "company_size": icp_info.get("company_size", "10-200 employees"),
                "titles": icp_info.get("titles", ["VP", "Director"]),
                "geography": icp_info.get("geography", ["United States"]),
            },
            "channels": {
                "email": {
                    "enabled": True,
                    "provider": "draft",
                    "max_daily_sends": 50,
                    # The trained site is yours: autopilot may link to it.
                    "link_domains": [email_domain],
                },
            },
            "usage": {
                "max_daily_percent": 80,
                "heartbeat_interval_minutes": 15,
                "quiet_hours": {
                    "start": "22:00",
                    "end": "07:00",
                    "timezone": "America/New_York",
                },
            },
            "brain": brain_section(self.brain_config),
        }


async def run_training(url: str, output: str = "sellbird.local.yaml", max_pages: int = 100):
    """Entry point for training."""
    trainer = Trainer()
    await trainer.train(url, output, max_pages=max_pages)


def main():
    """CLI entry point for training."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m sellbird.trainer <website-url> [max-pages]")
        print("")
        print("Examples:")
        print("  python -m sellbird.trainer https://example.com")
        print("  python -m sellbird.trainer https://example.com 200")
        sys.exit(1)

    url = _normalize_start_url(sys.argv[1])

    max_pages = 100
    if len(sys.argv) > 2:
        try:
            max_pages = max(1, int(sys.argv[2]))
        except ValueError:
            print(f"Invalid max-pages value: {sys.argv[2]!r} (expected a number)")
            sys.exit(1)

    asyncio.run(run_training(url, max_pages=max_pages))


if __name__ == "__main__":
    from sellbird.textio import utf8_stdio
    utf8_stdio()
    main()
