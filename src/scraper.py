import os
import re
import time
import logging
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from urllib.request import urlretrieve

from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
GRADE_PATTERN = re.compile(r"(PSA|CGC|Beckett|TAG)\s+(\d+)", re.IGNORECASE)

def init_driver():
    """
    Initialize a headless Selenium Microsoft Edge driver.
    Ensure msedgedriver is installed and in your PATH.
    """
    options = EdgeOptions()
    options.use_chromium = True
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Edge(options=options)
    return driver

def scrape_all_pages_selenium(base_url, output_dir="ENSF 544\Final-Project\data\cards", max_pages=95, delay=5):
    """
    Scrape card images and grade labels using Selenium (with Microsoft Edge) to render JavaScript.
    
    Args:
        base_url (str): URL for page 1.
        output_dir (str): Directory to save images.
        max_pages (int): Maximum number of pages to scrape.
        delay (int): Seconds to wait for the page to fully load.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    driver = init_driver()
    
    page = 1
    while page <= max_pages:
        # Update the page number in the URL
        parsed_url = list(urlparse(base_url))
        query_params = parse_qs(parsed_url[4])
        query_params["page"] = [str(page)]
        parsed_url[4] = urlencode(query_params, doseq=True)
        page_url = urlunparse(parsed_url)
        
        # logger.info(f"Scraping page {page}: {page_url}")
        driver.get(page_url)
        # Wait for JavaScript to load the dynamic content
        time.sleep(delay)
        
        # Get the rendered HTML from the page
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        # Select all <img> tags with the unique class for card images
        img_tags = soup.find_all("img", class_="fill max-h-full max-w-full rounded object-contain drop-shadow")
        if not img_tags:
            logger.info(f"No matching image tags found on page {page}. Ending scrape.")
            break
        
        for idx, img_tag in enumerate(img_tags):
            alt_text = img_tag.get("alt", "")
            if not alt_text:
                logger.debug("No alt text found; skipping this image.")
                continue
            
            # Extract the numeric grade from the alt text (e.g., "PSA 5")
            match = re.search(r"(PSA|CGC|Beckett|TAG|BGS)\s+(\d+)", alt_text, re.IGNORECASE)
            if not match:
                # logger.warning(f"No PSA grade found in alt text: '{alt_text}'")
                continue
            
            try:
                grade_num = int(match.group(2))
            except ValueError as e:
                logger.error(f"Error converting grade: {e}")
                continue

            img_url = img_tag.get("src", "")
            if not img_url.startswith("http"):
                img_url = urljoin(page_url, img_url)
            
            # Create a subdirectory for this grade if it doesn't exist
            grade_dir = os.path.join(output_dir, f"grade_{grade_num}")
            os.makedirs(grade_dir, exist_ok=True)
            
            
            img_filename = os.path.join(grade_dir, f"card_{page}_{idx}.jpg")
            try:
                urlretrieve(img_url, img_filename)
                # logger.info(f"Downloaded image: {img_filename} (Alt: '{alt_text}')")
            except Exception as e:
                logger.error(f"Error downloading image {img_url}: {e}")
        
        page += 1

    driver.quit()

if __name__ == "__main__":
    # Replace this base_url with the actual URL for page 1 of your target site.
    base_url = (
        "https://www.fanaticscollect.com/weekly-auction?%22%22category=Trading+Card+Games+%3E+Pok%C3%A9mon+(English),%22%22Trading+Card+Games+%3E+Pok%C3%A9mon+(Japanese),%22%22Trading+Card+Games+%3E+Pok%C3%A9mon+(Other+Languages)%22%22&type=WEEKLY&grade=0,10&page=1&category=Trading+Card+Games+%3E+Pok%C3%A9mon+(English),Trading+Card+Games+%3E+Pok%C3%A9mon+(Japanese)"
    )
    scrape_all_pages_selenium(base_url, output_dir="ENSF 544\Final-Project\data\cards", max_pages=95, delay=5)
