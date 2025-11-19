"""
Quick script to fetch Mark Daou data using credentials from .env
NO SAMPLE DATA - ONLY REAL API DATA
"""
import os
from dotenv import load_dotenv
from comprehensive_fetcher import ComprehensivePoliticalDataFetcher

# Load credentials from .env file
load_dotenv()

bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
username = os.getenv('POLITICIAN_USERNAME', 'DaouMark')

if not bearer_token:
    print("[ERROR] TWITTER_BEARER_TOKEN not found in .env file")
    exit(1)

print("[AUTH] Using Bearer Token from .env")
print(f"[TARGET] Username: {username}")
print()

# Fetch data
fetcher = ComprehensivePoliticalDataFetcher(bearer_token)
data = fetcher.fetch_comprehensive_data(username)

if data:
    print("\n" + "="*60)
    print("[SUCCESS] Data collected from X API")
    print("="*60)
    print("\n[NEXT STEP] Launch dashboard")
    print("   streamlit run enhanced_dashboard.py")
else:
    print("\n[ERROR] Failed to collect data")
    print("Check your credentials and username")

