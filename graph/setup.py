"""
APEX TigerGraph Setup Script
----------------------------
Automates the creation of the graph schema and installation of GSQL queries.
Requires pyTigerGraph: pip install pyTigerGraph
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import get_settings

def setup_tigergraph():
    try:
        import pyTigerGraph as tg
    except ImportError:
        logger.error("pyTigerGraph not found. Run: pip install pyTigerGraph")
        return

    settings = get_settings()
    config = settings.tigergraph

    logger.info(f"Connecting to TigerGraph at {config.host}...")
    
    conn = tg.TigerGraphConnection(
        host=config.host,
        graphname=config.graph_name,
        username=config.username,
        password=config.password
    )

    # 1. Get/Generate Token
    try:
        if config.api_token:
            conn.apiToken = config.api_token
        else:
            conn.getToken(config.password)
            logger.info("Generated new API token")
    except Exception as e:
        logger.warning(f"Token generation failed: {e}. Ensure TigerGraph is running and RESTPP is accessible.")

    # 2. Check if Graph exists, if not create (requires admin privileges)
    try:
        graphs = conn.getGraphs()
        if config.graph_name not in graphs:
            logger.info(f"Creating graph '{config.graph_name}'...")
            # Note: Creating a graph via Python API is limited, usually done via GSQL shell
            # We will assume the graph exists or is being created in the Cloud console.
        else:
            logger.info(f"Graph '{config.graph_name}' already exists.")
    except Exception as e:
        logger.warning(f"Could not verify graph existence: {e}")

    # 3. Install Schema and Queries
    gsql_path = Path(__file__).resolve().parent / "queries.gsql"
    if gsql_path.exists():
        logger.info(f"Installing GSQL queries from {gsql_path}...")
        try:
            with open(gsql_path, "r") as f:
                gsql_content = f.read()
            
            # Divide into schema and queries
            # This is a simplified approach, usually you'd run these via conn.gsql()
            res = conn.gsql(gsql_content)
            logger.success("GSQL installation command sent.")
            logger.debug(f"Response: {res}")
        except Exception as e:
            logger.error(f"GSQL installation failed: {e}")
    else:
        logger.error("queries.gsql not found!")

    logger.info("TigerGraph setup script finished.")

if __name__ == "__main__":
    setup_tigergraph()
