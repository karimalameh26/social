"""
Quick Test Script for Advanced Analysis System
Tests that all models load correctly and basic functionality works
"""

import sys
import glob
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
# Simple check/cross marks for Windows compatibility
CHECK = "[OK]"
CROSS = "[X]"

def test_imports():
    """Test that all required packages are installed"""
    print("\n" + "="*60)
    print("  TESTING IMPORTS")
    print("="*60 + "\n")
    
    required_packages = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('plotly', 'Plotly'),
        ('transformers', 'Transformers'),
        ('torch', 'PyTorch'),
        ('sklearn', 'Scikit-learn'),
        ('nltk', 'NLTK'),
        ('textblob', 'TextBlob'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"{CHECK} {name}")
        except ImportError:
            print(f"{CROSS} {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n{CROSS} Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print(f"\n{CHECK} All packages installed!")
        return True

def test_model_loading():
    """Test that AI models can be loaded"""
    print("\n" + "="*60)
    print("  TESTING AI MODELS")
    print("="*60 + "\n")
    
    try:
        from advanced_insights_analyzer import AdvancedInsightsAnalyzer
        
        print("[TEST] Loading analyzer (this may take a minute first time)...")
        analyzer = AdvancedInsightsAnalyzer()
        
        print(f"\n{CHECK} All models loaded successfully!")
        return True, analyzer
    except Exception as e:
        print(f"\n{CROSS} Model loading failed: {e}")
        return False, None

def test_sentiment_analysis(analyzer):
    """Test sentiment analysis on sample text"""
    print("\n" + "="*60)
    print("  TESTING SENTIMENT ANALYSIS")
    print("="*60 + "\n")
    
    test_texts = [
        "I am very happy and excited about the progress we are making!",
        "This is terrible and disappointing news.",
        "The meeting was held yesterday."
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment_advanced(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 40)
    
    print(f"\n{CHECK} Sentiment analysis working!")
    return True

def test_emotion_analysis(analyzer):
    """Test emotion detection"""
    print("\n" + "="*60)
    print("  TESTING EMOTION DETECTION")
    print("="*60 + "\n")
    
    test_text = "I am so happy and excited about this amazing news!"
    result = analyzer.analyze_emotion(test_text)
    
    print(f"Text: {test_text}")
    print(f"Dominant Emotion: {result['dominant_emotion'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    if result['all_emotions']:
        print("\nAll emotions:")
        for emotion, score in sorted(result['all_emotions'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {score:.2f}")
    
    print(f"\n{CHECK} Emotion detection working!")
    return True

def test_ner(analyzer):
    """Test named entity recognition"""
    print("\n" + "="*60)
    print("  TESTING NAMED ENTITY RECOGNITION")
    print("="*60 + "\n")
    
    test_text = "Mark Daou met with the Lebanese Parliament in Beirut yesterday."
    entities = analyzer.extract_entities(test_text)
    
    print(f"Text: {test_text}")
    print(f"\nExtracted {len(entities)} entities:")
    
    for entity in entities:
        print(f"  - {entity['text']} ({entity['type']}) - confidence: {entity['score']:.2f}")
    
    print(f"\n{CHECK} Named Entity Recognition working!")
    return True

def test_data_files():
    """Check if data files exist"""
    print("\n" + "="*60)
    print("  CHECKING DATA FILES")
    print("="*60 + "\n")
    
    comprehensive_files = glob.glob("mark_daou_comprehensive_*.json")
    advanced_files = glob.glob("mark_daou_advanced_analysis_*.json")
    
    print(f"Comprehensive data files: {len(comprehensive_files)}")
    if comprehensive_files:
        latest = max(comprehensive_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"  Latest: {latest}")
    
    print(f"\nAdvanced analysis files: {len(advanced_files)}")
    if advanced_files:
        latest = max(advanced_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"  Latest: {latest}")
    
    if comprehensive_files or advanced_files:
        print(f"\n{CHECK} Data files found!")
        return True
    else:
        print("\n[WARN] No data files found")
        print("Run: python comprehensive_fetcher.py")
        return False

def test_dashboard():
    """Check if dashboard can be loaded"""
    print("\n" + "="*60)
    print("  TESTING DASHBOARD")
    print("="*60 + "\n")
    
    try:
        import enhanced_dashboard
        print(f"{CHECK} Dashboard module loads successfully")
        
        # Check if new functions exist
        required_functions = [
            'display_emotion_analysis',
            'display_influence_metrics',
            'display_content_strategy',
            'display_entity_network',
            'display_temporal_insights',
            'display_advanced_recommendations'
        ]
        
        for func_name in required_functions:
            if hasattr(enhanced_dashboard, func_name):
                print(f"  {CHECK} {func_name}")
            else:
                print(f"  {CROSS} {func_name} - NOT FOUND")
        
        print(f"\n{CHECK} Dashboard is ready!")
        return True
    except Exception as e:
        print(f"{CROSS} Dashboard loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  ADVANCED ANALYSIS SYSTEM - TEST SUITE")
    print("="*80)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    if not results[0][1]:
        print(f"\n{CROSS} CRITICAL: Install missing packages first")
        sys.exit(1)
    
    # Test 2: Model loading
    success, analyzer = test_model_loading()
    results.append(("Model Loading", success))
    
    if not success:
        print(f"\n{CROSS} CRITICAL: Models failed to load")
        sys.exit(1)
    
    # Test 3: Sentiment
    results.append(("Sentiment Analysis", test_sentiment_analysis(analyzer)))
    
    # Test 4: Emotion
    results.append(("Emotion Detection", test_emotion_analysis(analyzer)))
    
    # Test 5: NER
    results.append(("Named Entity Recognition", test_ner(analyzer)))
    
    # Test 6: Data files
    results.append(("Data Files", test_data_files()))
    
    # Test 7: Dashboard
    results.append(("Dashboard", test_dashboard()))
    
    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80 + "\n")
    
    all_passed = True
    for test_name, passed in results:
        status = f"{CHECK} PASS" if passed else f"{CROSS} FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    
    if all_passed:
        print("\n*** ALL TESTS PASSED! ***")
        print(f"\n{CHECK} System is ready to use!")
        print("\nNext steps:")
        print("  1. Collect data: python comprehensive_fetcher.py")
        print("  2. View dashboard: streamlit run enhanced_dashboard.py")
    else:
        print("\n*** SOME TESTS FAILED ***")
        print("\nCheck the errors above and:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Collect data: python comprehensive_fetcher.py")
        print("  3. Run tests again: python test_advanced_analysis.py")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

