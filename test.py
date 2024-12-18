import asyncio
import logging
from pyppeteer import launch
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class URLResolver:
    def __init__(self, timeout: int = 10):
        """
        Initialize URL resolver with configurable timeout
        
        Args:
            timeout (int): Maximum time to wait for page load in seconds
        """
        self.timeout = timeout

    async def resolve_redirect(self, url: str) -> Optional[str]:
        """
        Resolve redirect URL using Puppeteer
        
        Args:
            url (str): URL to resolve
        
        Returns:
            Optional[str]: Final resolved URL or None if resolution fails
        """
        browser = None
        try:
            # Launch browser in headless mode
            browser = await launch(headless=True, args=['--no-sandbox'])
            page = await browser.newPage()
            
            # Set a higher timeout for navigation
            await page.setDefaultNavigationTimeout(self.timeout * 1000)
            
            # Navigate to the URL
            response = await page.goto(url, {'waitUntil': 'networkidle0'})
            
            # Get the final URL after all redirects
            final_url = page.url
            
            return final_url
        
        except Exception as e:
            logging.error(f"Error resolving URL {url}: {e}")
            return None
        
        finally:
            # Ensure browser is closed
            if browser:
                await browser.close()

    async def resolve_urls_concurrently(self, urls: List[str]) -> List[str]:
        """
        Resolve multiple URLs concurrently
        
        Args:
            urls (List[str]): List of URLs to resolve
        
        Returns:
            List[str]: List of resolved URLs
        """
        # Use asyncio to run resolution concurrently
        tasks = [self.resolve_redirect(url) for url in urls]
        resolved_urls = await asyncio.gather(*tasks)
        
        # Filter out None values
        return [url for url in resolved_urls if url is not None]

def resolve_urls(urls: List[str], timeout: int = 10) -> List[str]:
    """
    Synchronous wrapper for URL resolution
    
    Args:
        urls (List[str]): List of URLs to resolve
        timeout (int): Timeout for each URL resolution
    
    Returns:
        List[str]: List of resolved URLs
    """
    resolver = URLResolver(timeout)
    return asyncio.get_event_loop().run_until_complete(
        resolver.resolve_urls_concurrently(urls)
    )

# Example usage
if __name__ == "__main__":
    redirect_urls = [
        "https://news.google.com/rss/articles/CBMia0FVX3lxTE5EZFdNWTktNlphc3lNV3cxbmktcS1KdTVLUmVmZ2pESnBWTk9QZl9XSEZYbDF3YUdhelpoeXZ3RTJ6dFJoZFQzaElLSWdmbFZwblpySUJtVXBKSm9aN0xxUjMxTFVYeFUyVDR3?oc=5",
        "https://news.google.com/rss/articles/CBMimwFBVV95cUxNTEtTMlFITEJSYkx4Nm9mN2poMW9KR0ppWWNDODVNSUhnVzlFeU1mUWNvalZlUWZ3NFZoZmhiRm9DXzlHUXQ2c2xpLUZKenN3ckxQZXQ3TGVPbkNDMXpuN1VRcTAwTFVLLTBTU1JUVnQ1dWVCOGlfY0xTMkh0MzUwdmRkcVIyVkNUelVBb3JTblY3T0l6ZDN4TFpmRQ?oc=5"
        # Add your redirect URLs here
    ]
    
    resolved_urls = resolve_urls(redirect_urls)
    for url in resolved_urls:
        print(f"Resolved URL: {url}")