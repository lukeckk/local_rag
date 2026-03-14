"""
MOM FAQ scraper using Crawl4AI v0.8.0.

Fetches each target URL, converts HTML tables/lists to clean Markdown
using the PruningContentFilter, and writes one .md file per page
to the output directory.
"""

import asyncio
import os
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from config import MOM_URLS, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _url_to_filename(url: str) -> str:
    """Convert a URL into a safe, descriptive filename."""
    slug = re.sub(r"https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_")
    return f"{slug}.md"


def _build_browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--no-sandbox"],
    )


def _build_run_config() -> CrawlerRunConfig:
    """
    Uses raw markdown without pruning — MOM pages are JS-heavy and
    the pruning filter strips too aggressively on navigation-heavy layouts.
    """
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(),
        wait_until="networkidle",
        page_timeout=30000,
        delay_before_return_html=2.0,
    )


async def scrape_url(
    crawler: AsyncWebCrawler,
    run_config: CrawlerRunConfig,
    url: str,
    output_dir: Path,
) -> bool:
    """Scrape a single URL and write clean Markdown to disk. Returns True on success."""
    logger.info(f"Scraping: {url}")
    try:
        result = await crawler.arun(url=url, config=run_config)

        if not result.success:
            logger.warning(f"Failed to crawl {url}: {result.error_message}")
            return False

        content = result.markdown.raw_markdown

        if not content or len(content.strip()) < 50:
            logger.warning(f"Skipping {url}: content too short after filtering")
            return False

        # Prepend metadata header so the embedding pipeline knows the source
        header = (
            f"---\n"
            f"source_url: {url}\n"
            f"scraped_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"---\n\n"
        )
        filename = output_dir / _url_to_filename(url)
        filename.write_text(header + content, encoding="utf-8")
        logger.info(f"Saved {len(content)} chars → {filename.name}")
        return True

    except Exception as exc:
        logger.error(f"Error scraping {url}: {exc}")
        return False


async def run_crawl(urls: list[str] | None = None, output_dir_override: str | None = None) -> dict:
    """
    Main entry point: crawl all (or a subset of) MOM URLs.
    Returns a summary dict with success/failure counts.
    """
    target_urls = urls or MOM_URLS
    out_dir = Path(output_dir_override or os.path.join(os.path.dirname(__file__), OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    browser_config = _build_browser_config()
    run_config = _build_run_config()

    success_count = 0
    fail_count = 0

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in target_urls:
            ok = await scrape_url(crawler, run_config, url, out_dir)
            if ok:
                success_count += 1
            else:
                fail_count += 1
            # Polite delay between requests
            await asyncio.sleep(2)

    summary = {
        "total": len(target_urls),
        "success": success_count,
        "failed": fail_count,
        "output_dir": str(out_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"Crawl complete: {summary}")
    return summary


if __name__ == "__main__":
    asyncio.run(run_crawl())
