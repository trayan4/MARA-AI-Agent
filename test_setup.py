"""
Simple test script to verify MARA setup.
Tests core components and basic functionality.
"""

import sys
from pathlib import Path

# Add mara to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*80)
    print("TEST 1: Imports")
    print("="*80)
    
    try:
        from config import settings
        print("✅ Config loaded")
        
        from tools.openai_client import get_openai_client
        print("✅ OpenAI client")
        
        from tools.chunking import chunk_text
        print("✅ Chunking utility")
        
        from tools.local_vector_store import get_vector_store
        print("✅ Vector store")
        
        from tools.python_executor import PythonExecutor
        print("✅ Python executor")
        
        from agents.planner import get_planner
        print("✅ Planner agent")
        
        from agents.rag import get_rag_agent
        print("✅ RAG agent")
        
        from agents.vision import get_vision_agent
        print("✅ Vision agent")
        
        from agents.data import get_data_agent
        print("✅ Data agent")
        
        from agents.web_search import get_web_search_agent
        print("✅ Web search agent")
        
        from agents.critic import get_critic_agent
        print("✅ Critic agent")
        
        from agents.report import get_report_agent
        print("✅ Report agent")
        
        from orchestrator.graph import get_mara_graph
        print("✅ Orchestrator graph")
        
        from api.main import app
        print("✅ FastAPI app")
        
        print("\n✅ All imports successful!")
        return True
    
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_chunking():
    """Test document chunking."""
    print("\n" + "="*80)
    print("TEST 2: Chunking")
    print("="*80)
    
    try:
        from tools.chunking import chunk_text
        
        text = """
        This is a test document.
        It has multiple paragraphs.
        
        This is the second paragraph.
        It should be chunked properly.
        """
        
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        
        print(f"Original text length: {len(text)} chars")
        print(f"Number of chunks: {len(chunks)}")
        print(f"First chunk: {chunks[0].text[:50]}...")
        
        print("\n✅ Chunking works!")
        return True
    
    except Exception as e:
        print(f"\n❌ Chunking failed: {e}")
        return False


def test_vector_store():
    """Test vector store."""
    print("\n" + "="*80)
    print("TEST 3: Vector Store")
    print("="*80)
    
    try:
        from tools.local_vector_store import get_vector_store
        from tools.chunking import chunk_text
        
        vector_store = get_vector_store()
        
        # Add a test document
        doc_content = "Python is a programming language. It is widely used for data science and AI."
        chunks = chunk_text(doc_content)
        
        success = vector_store.add_document(
            doc_id="test_doc_1",
            content=doc_content,
            chunks=chunks,
            metadata={"test": True}
        )
        
        if success:
            print("✅ Document added to vector store")
        
        # Search
        results = vector_store.search("What is Python?", top_k=3)
        print(f"✅ Search returned {len(results)} results")
        
        # Get stats
        stats = vector_store.get_stats()
        print(f"✅ Vector store stats: {stats['num_documents']} docs, {stats['num_chunks']} chunks")
        
        print("\n✅ Vector store works!")
        return True
    
    except Exception as e:
        print(f"\n❌ Vector store failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_agent():
    """Test RAG agent."""
    print("\n" + "="*80)
    print("TEST 4: RAG Agent")
    print("="*80)
    
    try:
        from agents.rag import get_rag_agent
        
        rag_agent = get_rag_agent()
        
        # Add a document
        success = rag_agent.add_document(
            doc_id="test_rag_doc",
            content="MARA is a multimodal agentic reasoning assistant. It can analyze documents, images, and data.",
            metadata={"source": "test"}
        )
        
        if success:
            print("✅ Document added via RAG agent")
        
        # Answer question
        result = rag_agent.answer_question("What is MARA?")
        
        print(f"✅ Question answered")
        print(f"   Answer: {result.answer[:100]}...")
        print(f"   Confidence: {result.confidence:.2f}")
        
        print("\n✅ RAG agent works!")
        return True
    
    except Exception as e:
        print(f"\n❌ RAG agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_planner():
    """Test planner agent."""
    print("\n" + "="*80)
    print("TEST 5: Planner Agent")
    print("="*80)
    
    try:
        from agents.planner import get_planner
        
        planner = get_planner()
        
        # Create a plan
        plan = planner.plan(
            query="Analyze the sales data and create a summary",
            context={"uploaded_files": ["sales.csv"]}
        )
        
        print(f"✅ Plan created with {len(plan.tasks)} tasks")
        print(f"   Reasoning: {plan.reasoning}")
        
        for task in plan.tasks:
            print(f"   - Task {task.task_id}: {task.agent_type}.{task.tool_name}")
        
        print("\n✅ Planner works!")
        return True
    
    except Exception as e:
        print(f"\n❌ Planner failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration."""
    print("\n" + "="*80)
    print("TEST 6: Configuration")
    print("="*80)
    
    try:
        from config import settings
        
        print(f"✅ LLM model: {settings.llm.model}")
        print(f"✅ Embedding model: {settings.embeddings.model}")
        print(f"✅ Vector store type: {settings.vector_store.type}")
        print(f"✅ API port: {settings.api.port}")
        
        # Check if OpenAI key is set
        import os
        if os.getenv("OPENAI_API_KEY"):
            print(f"✅ OpenAI API key is set")
        else:
            print(f"⚠️  OpenAI API key NOT set (tests will be limited)")
        
        print("\n✅ Configuration loaded!")
        return True
    
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("MARA SYSTEM TEST SUITE")
    print("="*80)
    
    tests = [
        ("Configuration", test_config),
        ("Imports", test_imports),
        ("Chunking", test_chunking),
        ("Vector Store", test_vector_store),
        ("RAG Agent", test_rag_agent),
        ("Planner", test_planner),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! MARA is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)