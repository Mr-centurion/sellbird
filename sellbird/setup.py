"""Interactive setup wizard — walks you through everything Sellbird needs."""

import asyncio
import getpass
import logging
import os
import shutil
from pathlib import Path

import yaml

from pydantic import ValidationError

from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.state import StateManager

logger = logging.getLogger("sellbird.setup")

from sellbird.paths import PROJECT_ROOT
from sellbird.textio import read_user_text
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
CONFIG_FILE = PROJECT_ROOT / "sellbird.yaml"


def _print_sellbird(message: str):
    """Print a message as Sellbird."""
    print(f"\n  Sellbird: {message}")


def _print_step(step: int, total: int, title: str):
    """Print a step header."""
    print(f"\n{'─'*60}")
    print(f"  Step {step}/{total}: {title}")
    print(f"{'─'*60}")


def _ask(prompt: str, default: str = "", required: bool = True, secret: bool = False) -> str:
    """Ask the user a question. secret=True hides input (passwords, API keys)."""
    suffix = f" [{default}]" if default else ""
    reader = getpass.getpass if secret else input
    while True:
        try:
            answer = reader(f"\n  → {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Setup cancelled.")
            raise SystemExit(1)

        if not answer and default:
            return default
        if not answer and required:
            print("    (required — please enter a value)")
            continue
        return answer


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        answer = input(f"\n  → {prompt}{suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Setup cancelled.")
        raise SystemExit(1)

    if not answer:
        return default
    return answer in ("y", "yes")


def _read_env_file() -> dict[str, str]:
    """The KEY=value pairs already in .env (comments and blank lines skipped)."""
    existing: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in read_user_text(ENV_FILE).splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                existing[key.strip()] = value.strip()
    return existing


def _config_file() -> Path:
    """The config Sellbird loads: sellbird.local.yaml (what `sellbird train` writes)
    wins over sellbird.yaml when it exists, as in config._find_config_file.
    Setup reads and writes this file, so its answers are the ones Sellbird uses."""
    local = CONFIG_FILE.with_name("sellbird.local.yaml")
    return local if local.exists() else CONFIG_FILE


def _merge_answers(existing: dict, answers: dict) -> None:
    """Lay manual setup answers over existing config, skipping blank answers so
    they never wipe values already in the file."""
    for key, value in answers.items():
        if isinstance(value, dict) and isinstance(existing.get(key), dict):
            _merge_answers(existing[key], value)
        elif value not in ("", [], {}, None):
            existing[key] = value


def _product_configured() -> bool:
    """Whether the config names a real company and product, not the shipped
    placeholders (the same test the dashboard uses)."""
    try:
        data = yaml.safe_load(read_user_text(_config_file()))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    persona = data.get("persona") if isinstance(data.get("persona"), dict) else {}
    product = data.get("product") if isinstance(data.get("product"), dict) else {}
    return (persona.get("company") not in (None, "", "Your Company")
            and product.get("name") not in (None, "", "Your Product"))


async def _test_brain(brain: Brain) -> bool:
    """Test that the configured brain answers."""
    response = await brain.think(
        "Respond with exactly: SELLBIRD_READY",
        session_id="sellbird-setup-test",
        agent="trainer",
        task="setup_test",
    )
    return "SELLBIRD_READY" in response


def _ask_brain_config() -> BrainConfig:
    """Configure Ollama as the local brain."""
    _print_sellbird("I think locally with Ollama (default model: qwen3:1.7b):")
    model = _ask("Model (leave blank for qwen3:1.7b)", default="qwen3:1.7b", required=False)
    try:
        return BrainConfig(provider="ollama", model=model or "qwen3:1.7b")
    except ValidationError:
        print("    (invalid model name — using qwen3:1.7b)")
        return BrainConfig(provider="ollama", model="qwen3:1.7b")


class SetupWizard:
    def __init__(self):
        self.env_vars: dict[str, str] = {}
        self.config: dict = {}
        self.brain_config = BrainConfig()
        # Whether _write_config saved the config, for the summary.
        self.config_saved = False
        # True when step was answered by hand: self.config then holds
        # persona/product/icp that are not in sellbird.yaml yet.
        self.manual_product = False

    async def run(self):
        """Run the full interactive setup."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   Sellbird Setup Wizard                                    ║
║   Let's get you closing deals.                           ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝""")

        _print_sellbird("Hey. I'm Sellbird. Let's get me set up so I can start")
        _print_sellbird("closing deals for you. I'll walk you through everything.")
        _print_sellbird("This takes about 5 minutes.\n")

        total_steps = 3

        # Step 1: Check prerequisites
        _print_step(1, total_steps, "Checking Prerequisites")
        await self._check_prerequisites()

        # Step 2: Train on product
        _print_step(2, total_steps, "Train Sellbird on Your Product")
        await self._setup_product()

        # Step 3: Configure behavior
        _print_step(3, total_steps, "Configure Behavior")
        self._setup_behavior()

        # Write everything
        self._write_env()
        self._write_config()

        # Final summary
        await self._print_summary()

    async def _check_prerequisites(self):
        """Pick the brain provider and verify Ollama is reachable."""
        self.brain_config = _ask_brain_config()
        await self._check_ollama()

    async def _check_ollama(self):
        print("\n  Checking Ollama service...", end=" ", flush=True)
        state = StateManager()
        await state.init_db()
        brain = Brain(state, self.brain_config)

        from sellbird.ollama_runtime import is_ollama_ready
        url = self.brain_config.ollama_url or "http://localhost:11434"
        model = self.brain_config.model or "qwen3:1.7b"
        ready, detail = await is_ollama_ready(url, model)
        if not ready:
            print("✗ Not reachable")
            _print_sellbird(f"Could not connect to Ollama: {detail}")
            _print_sellbird(f"Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull {model}`).")
            if not _ask_yes_no("Continue anyway?", default=False):
                raise SystemExit(1)
            return

        print("✓ Connected")
        print("  Testing model response...", end=" ", flush=True)
        try:
            if await _test_brain(brain):
                print("✓ Working")
            else:
                print("✗ Unexpected response")
                _print_sellbird("Model responded but not with the expected text.")
                if not _ask_yes_no("Continue anyway?", default=False):
                    raise SystemExit(1)
        except Exception as e:
            print(f"✗ Error: {e}")
            _print_sellbird(f"Test call failed: {e}")
            if not _ask_yes_no("Continue anyway?", default=False):
                raise SystemExit(1)

        _print_sellbird("Prerequisites look good. Let's keep going.")

    async def _setup_product(self):
        """Train Sellbird on the product — either via URL or manual entry."""
        _print_sellbird("Now the important part — I need to learn about what you're selling.")
        _print_sellbird("I can either crawl your website and figure it out myself,")
        _print_sellbird("or you can tell me the basics manually.\n")

        method = _ask(
            "Train from website URL or enter manually? (url/manual)",
            default="url",
        )

        if method.lower().startswith("u"):
            url = _ask("Your website URL (e.g. https://yourcompany.com)")
            if not url.startswith("http"):
                url = f"https://{url}"

            print(f"\n  Crawling {url} and training...", flush=True)
            try:
                from sellbird.trainer import Trainer
                state = StateManager()
                await state.init_db()
                trainer = Trainer(state=state, brain_config=self.brain_config)
                config_path = await trainer.train(url)
                _print_sellbird(f"Trained. Saved product knowledge to {config_path}.")
                _print_sellbird("You can fine-tune objection handling and ICP in sellbird.yaml anytime.")
            except Exception as e:
                _print_sellbird(f"Crawling failed: {e}")
                _print_sellbird("No problem — let's do the quick manual setup instead.")
                await self._manual_product_setup()
        else:
            await self._manual_product_setup()

    async def _manual_product_setup(self):
        """Manual product data entry."""
        _print_sellbird("Tell me the basics and I'll fill in the rest.\n")

        company = _ask("Company name")
        product = _ask("Product name")
        description = _ask("What does it do in one sentence?")
        pricing = _ask("Pricing model (e.g. '$49/mo', 'from $500', 'contact for pricing')", default="Contact for pricing")

        _print_sellbird("What are 1-3 key benefits customers get?")
        benefits = []
        for i in range(3):
            b = _ask(f"Benefit {i+1} (or Enter to finish)", required=(i == 0))
            if not b:
                break
            benefits.append(b)

        _print_sellbird("Now tell me about who you're targeting:")
        industries_raw = _ask("Target industries (comma-separated)", default="Technology, Software")
        industries = [i.strip() for i in industries_raw.split(",") if i.strip()]

        titles_raw = _ask("Target job titles (comma-separated)", default="CEO, Founder, VP Sales, Head of Growth")
        titles = [t.strip() for t in titles_raw.split(",") if t.strip()]

        company_size = _ask("Target company size", default="10-200 employees")
        geography_raw = _ask("Target geography (comma-separated)", default="United States")
        geography = [g.strip() for g in geography_raw.split(",") if g.strip()]

        _print_sellbird("What are we offering in the first email?")
        print("  The best cold emails offer something specific and free upfront")
        print("  (e.g., 'a 5-minute video teardown of your pricing page').")
        primary_offer = _ask("Main offer (what you ultimately want, e.g. 'a 15-minute demo')", default="a 15-minute intro call")
        entry_offer = _ask("Entry offer (free/low-friction, e.g. 'a quick audit')", default="")

        goal_input = _ask("Call-to-action goal (book_call / start_trial / get_reply)", default="book_call")
        goal_map = {"1": "book_call", "2": "start_trial", "3": "get_reply"}
        goal = goal_map.get(goal_input, goal_input)
        booking_method = "calendar_link"
        booking_url = ""
        meeting_duration = "15 minutes"
        meeting_owner = ""

        if goal == "book_call":
            booking_url = _ask("Your booking/calendar URL (Calendly, Cal.com, etc.)", default="", required=False)
            meeting_duration = _ask("Default meeting duration", default="15 minutes")
            meeting_owner = _ask("Meeting owner name (if different from sender)", default="", required=False)

        _print_sellbird("Who should outreach emails come from?")
        persona_name = _ask("Sender name", default="Alex")
        persona_email = _ask("Sender email", default=f"{persona_name.lower()}@{company.lower().replace(' ', '')}.com")
        persona_role = _ask("Sender role/title", default="Business Development")

        self.config = {
            "persona": {
                "name": persona_name,
                "company": company,
                "role": persona_role,
                "email": persona_email,
                "linkedin": "",
                "tone": "professional, consultative, confident",
            },
            "product": {
                "name": product,
                "description": description,
                "pricing": pricing,
                "key_benefits": benefits or ["Benefit 1"],
                "objection_responses": {},
                "offer": {
                    "primary": primary_offer or "",
                    "entry": entry_offer or "",
                    "goal": goal,
                    "booking_method": booking_method,
                    "booking_url": booking_url,
                    "meeting_duration": meeting_duration,
                    "meeting_owner": meeting_owner or "",
                },
            },
            "icp": {
                "industries": industries or ["Technology"],
                "company_size": company_size,
                "titles": titles or ["VP", "Director"],
                "geography": geography,
            },
            "channels": {
                "email": {
                    "enabled": True,
                    "provider": "draft",
                    "max_daily_sends": 50,
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
        }

        self.manual_product = True
        _print_sellbird("Got it. I have everything I need about your product.")

    def _setup_behavior(self):
        """Configure Sellbird's operational behavior."""
        _print_sellbird("Last thing — a few settings for how I operate.\n")

        # Usage limit
        print("  I track my own usage so I don't eat your whole daily quota.")
        usage_pct = _ask(
            "Max % of daily usage I can consume",
            default="80",
        )
        try:
            usage_pct = min(100.0, max(1.0, float(usage_pct)))
        except ValueError:
            print("    (didn't catch that — using 80)")
            usage_pct = 80.0

        # Heartbeat interval
        print("\n  I wake up on a schedule to check for work.")
        interval = _ask(
            "How often should I check for work (minutes)?",
            default="15",
        )
        try:
            interval = max(1, int(interval))
        except ValueError:
            print("    (didn't catch that — using 15)")
            interval = 15

        # Quiet hours
        print("\n  I won't send emails or do outreach during quiet hours.")
        quiet_start = _ask("Quiet hours start (24h format)", default="22:00")
        quiet_end = _ask("Quiet hours end (24h format)", default="07:00")
        timezone = _ask("Your timezone", default="America/New_York")

        # Daily send limit
        daily_sends = _ask(
            "Max emails to send per day",
            default="50",
        )
        try:
            daily_sends = max(1, int(daily_sends))
        except ValueError:
            print("    (didn't catch that — using 50)")
            daily_sends = 50

        # Update config
        if self.config:
            self.config["usage"] = {
                "max_daily_percent": usage_pct,
                "heartbeat_interval_minutes": interval,
                "quiet_hours": {
                    "start": quiet_start,
                    "end": quiet_end,
                    "timezone": timezone,
                },
            }
            if "channels" in self.config and "email" in self.config["channels"]:
                self.config["channels"]["email"]["max_daily_sends"] = daily_sends

        _print_sellbird("Perfect. All configured.")

    def _write_env(self):
        """Write credentials to .env file, preserving any already in the file."""
        env_vars = dict(_read_env_file())
        for k, v in self.env_vars.items():
            if v != "":
                env_vars[k] = v

        if not env_vars:
            return

        lines = [
            "# Sellbird environment variables",
            "# Generated by setup wizard",
            "",
        ]
        for key, val in env_vars.items():
            lines.append(f"{key}={val}")

        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"Wrote {ENV_FILE}")

    def _write_config(self):
        """Write sellbird.yaml."""
        from sellbird.trainer import brain_section

        config_file = _config_file()
        if self.config:
            self.config["brain"] = brain_section(self.brain_config)
        if self.config and not config_file.exists():
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Wrote {config_file}")
            self.config_saved = True
        elif self.config:
            # Config already exists (shipped or from trainer): update behavior
            # settings, plus product answers when they were entered by hand
            try:
                existing = yaml.safe_load(read_user_text(config_file)) or {}
                if self.manual_product:
                    for section in ("persona", "product", "icp"):
                        if section in self.config:
                            if not isinstance(existing.get(section), dict):
                                existing[section] = {}
                            _merge_answers(existing[section], self.config[section])
                existing["usage"] = self.config.get("usage", existing.get("usage", {}))
                existing["brain"] = self.config["brain"]
                if "channels" in self.config:
                    existing.setdefault("channels", {})
                    if "email" in self.config["channels"]:
                        existing["channels"].setdefault("email", {})
                        existing["channels"]["email"]["max_daily_sends"] = (
                            self.config["channels"]["email"].get("max_daily_sends", 50)
                        )
                text = yaml.dump(existing, default_flow_style=False, sort_keys=False)
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(text)
                self.config_saved = True
            except Exception as e:
                _print_sellbird(f"Couldn't save your settings to {config_file.name} ({e}).")
                _print_sellbird("Nothing in it was changed. Fix the file and re-run setup.")

    async def _print_summary(self):
        """Print final summary and next steps."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   Setup Complete!                                        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

        checks = {
            "Ollama (local brain)": True,
            "Product trained": _product_configured(),
        }

        for name, ok in checks.items():
            status = "✓" if ok else "–"
            print(f"  {status} {name}")

        print(f"""
  Files created:
    - .env (your credentials)
    - {_config_file().name} (your configuration)

  To start Sellbird:
    sellbird run          (or: python -m sellbird)

  To re-train on a different product:
    sellbird train https://newproduct.com

  To re-run this setup:
    sellbird setup
""")


async def run_setup():
    wizard = SetupWizard()
    await wizard.run()


def main():
    wizard = SetupWizard()
    asyncio.run(wizard.run())


if __name__ == "__main__":
    main()
