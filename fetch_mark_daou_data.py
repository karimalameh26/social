"""
Fetch Mark Daou Data - Auto-configured with .env credentials
NO SAMPLE DATA - ONLY REAL API DATA
"""
import os
from dotenv import load_dotenv
from comprehensive_fetcher import ComprehensivePoliticalDataFetcher

# Load credentials from .env file
load_dotenv()

def main():
    print("\n" + "="*60)
    print("  Mark Daou Data Fetcher")
    print("  Using credentials from .env file")
    print("  NO SAMPLE DATA - ONLY REAL API DATA")
    print("="*60 + "\n")
    
    # Get Bearer Token from .env
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    
    if not bearer_token:
        print("❌ ERROR: Bearer Token not found in .env file")
        print("   Please check your .env file exists and contains:")
        print("   TWITTER_BEARER_TOKEN=your_token")
        exit(1)
    
    print("✅ Credentials loaded from .env")
    print()
    
    # Get username
    username = input("Enter Mark Daou's X username (e.g., markdaou_lb): ").strip()
    
    if not username:
        print("\n❌ Error: Username is required")
        exit(1)
    
    # Remove @ if included
    username = username.lstrip('@')
    
    print(f"\n🚀 Fetching REAL data from X API for @{username}...")
    print("   This will use multiple API endpoints")
    print("   All data is REAL - NO SAMPLE DATA\n")
    
    # Create fetcher and collect data
    fetcher = ComprehensivePoliticalDataFetcher(bearer_token)
    data = fetcher.fetch_comprehensive_data(username)
    
    if data:
        print("\n" + "="*60)
        print("✅ SUCCESS! Real data collected from X API")
        print("="*60)
        print("\n📊 Next step: Launch the dashboard")
        print("   streamlit run enhanced_dashboard.py")
        print()
    else:
        print("\n" + "="*60)
        print("❌ FAILED: Could not collect data")
        print("="*60)
        print("\nPossible issues:")
        print("  - Check username is correct")
        print("  - Verify API credentials in .env")
        print("  - Check internet connection")
        print("  - Review API rate limits")
        print()

if __name__ == "__main__":
    main()

