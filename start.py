#!/usr/bin/env python3
"""
Quick start script for MARA.
Checks prerequisites and starts the API server.
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_env_file():
    """Check if .env file exists."""
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  .env file not found")
        print("   Creating from .env.example...")
        
        example_path = Path(".env.example")
        if example_path.exists():
            import shutil
            shutil.copy(example_path, env_path)
            print("✅ Created .env file")
            print("   ⚠️  IMPORTANT: Add your OPENAI_API_KEY to .env")
            return False
        else:
            print("❌ .env.example not found")
            return False
    
    # Check if API key is set
    with open(env_path) as f:
        content = f.read()
        if "your_openai_api_key_here" in content or not any("OPENAI_API_KEY" in line and "=" in line for line in content.split('\n')):
            print("⚠️  OpenAI API key not set in .env")
            print("   Please edit .env and add your API key")
            return False
    
    print("✅ .env file configured")
    return True


def check_dependencies():
    """Check if dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import openai
        import sentence_transformers
        import langgraph
        print("✅ Core dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False


def create_directories():
    """Create required directories."""
    dirs = ["data", "data/uploads", "data/outputs", "data/logs"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Data directories created")
    return True


def run_tests():
    """Run quick tests."""
    print("\n🧪 Running quick tests...")
    try:
        from config import settings
        from tools.chunking import chunk_text
        
        # Quick test
        text = "This is a test."
        chunks = chunk_text(text)  # Use default settings from config
        
        if len(chunks) > 0:
            print("✅ Quick test passed")
            return True
        else:
            print("❌ Test failed")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def start_server():
    """Start the FastAPI server."""
    print("\n🚀 Starting MARA API server...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   Press CTRL+C to stop\n")
    
    try:
        import uvicorn
        from api.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 MARA server stopped")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    print("="*80)
    print("MARA - Multimodal Agentic Reasoning Assistant")
    print("="*80)
    print()
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Dependencies", check_dependencies),
        ("Directories", create_directories),
        ("Quick Test", run_tests),
    ]
    
    all_passed = True
    for name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("   For help, see README.md or run: python test_setup.py")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ All checks passed!")
    print("="*80)
    
    # Ask to start server
    response = input("\nStart MARA server? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        start_server()
    else:
        print("\n👋 To start server later, run: python start.py")
        print("   Or: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()