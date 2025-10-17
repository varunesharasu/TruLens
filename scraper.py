import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords')

class NewsScraper:
    def __init__(self):
        self.sources = [
            {
                "url": "https://timesofindia.indiatimes.com/",
                "name": "Times of India"
            },
            {
                "url": "https://www.thehindu.com/",
                "name": "The Hindu"
            }
        ]
        self.news_data_file = "news_data.json"
        self.load_existing_news()

    def load_existing_news(self):
        if os.path.exists(self.news_data_file):
            with open(self.news_data_file, 'r', encoding='utf-8') as f:
                self.news_data = json.load(f)
        else:
            self.news_data = {}

    def save_news_data(self):
        with open(self.news_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.news_data, f, ensure_ascii=False, indent=2)

    def scrape_news(self):
        total_new_articles = 0

        for source in self.sources:
            try:
                print(f"Scraping from {source['name']}...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                response = requests.get(source['url'], headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Find news articles - different selectors for different sources
                news_articles = []

                if source['name'] == 'Times of India':
                    selectors = [
                        'div.news-card',
                        'div.article',
                        'div.top-story',
                        'div.list-item',
                        'a[href*="/articleshow/"]'
                    ]
                elif source['name'] == 'The Hindu':
                    selectors = [
                        'div.article',
                        'div.story-card',
                        'div.element',
                        'h3.title',
                        'a[href*="/news/"]'
                    ]

                for selector in selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        title = self.extract_title(element)
                        if title and len(title.strip()) > 10:  # Filter out very short titles
                            news_articles.append({
                                'title': title.strip(),
                                'url': self.extract_url(element, source['url']),
                                'timestamp': datetime.now().isoformat(),
                                'source': source['name']
                            })

                # Remove duplicates and add new articles
                new_articles = 0
                for article in news_articles:
                    article_key = f"{article['source'].lower()}_{article['title'].lower().replace(' ', '')}"
                    if article_key not in self.news_data:
                        self.news_data[article_key] = article
                        new_articles += 1

                if new_articles > 0:
                    print(f"Scraped {new_articles} new articles from {source['name']}")
                    total_new_articles += new_articles
                else:
                    print(f"No new articles found from {source['name']}")

            except Exception as e:
                print(f"Error scraping {source['name']}: {e}")

        if total_new_articles > 0:
            self.save_news_data()
            print(f"Total: Scraped {total_new_articles} new articles. Total articles in database: {len(self.news_data)}")
        else:
            print("No new articles found from any source.")

    def extract_title(self, element):
        # Try different title selectors
        title_selectors = ['h2', 'h3', 'h4', '.title', '.headline', 'a']
        for selector in title_selectors:
            title_elem = element.select_one(selector)
            if title_elem:
                return title_elem.get_text().strip()
        return element.get_text().strip() if element.get_text() else None

    def extract_url(self, element, base_url):
        link = element.find('a') or element
        if link and link.get('href'):
            href = link['href']
            if href.startswith('/'):
                return base_url.rstrip('/') + href
            elif href.startswith('http'):
                return href
        return base_url

    def preprocess_text(self, text):
        text = re.sub(r'\W', ' ', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def check_news_exists(self, user_input):
        # Preprocess user input to extract keywords
        user_keywords = set(self.preprocess_text(user_input).split())

        for article_key, article in self.news_data.items():
            article_keywords = set(self.preprocess_text(article['title']).split())

            # Calculate keyword overlap percentage
            if article_keywords:
                matched_keywords = user_keywords.intersection(article_keywords)
                overlap = len(matched_keywords) / len(article_keywords)
                if overlap >= 0.7:  # 70% keyword overlap
                    return True, article, matched_keywords
        return False, None, set()

    def calculate_similarity(self, text1, text2):
        # Simple Jaccard similarity for basic matching
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0

# Global scraper instance
scraper = NewsScraper()

def start_scraping_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=scraper.scrape_news, trigger="interval", minutes=30)
    scheduler.start()

    # Initial scrape
    scraper.scrape_news()

    print("News scraping scheduler started. Will scrape every 30 minutes.")

if __name__ == "__main__":
    start_scraping_scheduler()
    # Keep the script running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping scraper...")