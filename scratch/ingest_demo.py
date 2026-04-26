"""
APEX Ingestion Demo Script
--------------------------
Demonstrates the ingestion of a document (README.md) into the self-improving graph.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.ingestion_service import get_ingestion_service

async def main():
    service = get_ingestion_service()
    
    # Read README
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    if not readme_path.exists():
        print("README.md not found!")
        return
        
    text = readme_path.read_text(encoding="utf-8")
    
    print(f"Ingesting README.md ({len(text)} chars)...")
    
    try:
        stats = await service.ingest_text(text, "README.md")
        print("\nIngestion Complete!")
        print(f"Status: SUCCESS")
        print(f"Chunks: {stats['chunks_count']}")
        print(f"Entities: {stats['entities_count']}")
        print(f"Relationships: {stats['relationships_count']}")
    except Exception as e:
        print(f"\nIngestion Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
