"""
Test script for LambdaCast Agent with LangSmith tracing.

This script tests the agent with different user roles and scenarios.
All interactions will be traced in LangSmith for debugging and analysis.

Usage:
    python test_agent_langsmith.py
"""

import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables (including LangSmith config)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the agent graph
from agent_app.graph import graph


def print_separator(title: str):
    """Print a nice separator for console output."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_scenario(
    scenario_name: str,
    user_message: str,
    session: dict,
    task: bool = False
):
    """
    Test a single scenario with the agent.
    
    Args:
        scenario_name: Name of the test scenario
        user_message: The user's message to the agent
        session: Session dict with user_id, username, role
        task: Whether this is a task (for graph state)
    """
    print_separator(f"TEST: {scenario_name}")
    
    # Build initial state
    initial_state = {
        "messages": [{"role": "user", "content": user_message}],
        "task": task,
        "session": session,
    }
    
    print(f"👤 User: {session['username']} (Role: {session['role']}, ID: {session['user_id']})")
    print(f"💬 Message: {user_message}\n")
    
    try:
        # Invoke the agent (this will be traced in LangSmith)
        result = graph.invoke(initial_state)
        
        # Extract the agent's response
        if result and "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            response_content = getattr(last_message, "content", str(last_message))
            print(f"🤖 Agent Response:\n{response_content}\n")
        else:
            print("❌ No response from agent\n")
            
        return result
    
    except Exception as e:
        logger.exception(f"Error in scenario '{scenario_name}'")
        print(f"❌ Error: {str(e)}\n")
        return None


def main():
    """Run all test scenarios."""
    
    print_separator("LambdaCast Agent Testing with LangSmith")
    print(f"LangSmith Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print(f"Tracing Enabled: {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # =========================================================================
    # Test Scenario 1: Regular User - List Installations
    # =========================================================================
    test_scenario(
        scenario_name="User Lists Their Installations",
        user_message="Show me my solar installations",
        session={
            "user_id": 5,  # Replace with actual user ID from your database
            "username": "john_doe",
            "role": "user"
        }
    )
    
    # =========================================================================
    # Test Scenario 2: Regular User - Get Current Power
    # =========================================================================
    test_scenario(
        scenario_name="User Checks Current Power",
        user_message="What's my current power output?",
        session={
            "user_id": 5,
            "username": "john_doe",
            "role": "user"
        }
    )
    
    # =========================================================================
    # Test Scenario 3: Regular User - Get Installation Details
    # =========================================================================
    test_scenario(
        scenario_name="User Gets Installation Details",
        user_message="Tell me about installation ID 1",
        session={
            "user_id": 5,
            "username": "john_doe",
            "role": "user"
        }
    )
    
    # =========================================================================
    # Test Scenario 4: Regular User - Forecast
    # =========================================================================
    test_scenario(
        scenario_name="User Requests Forecast",
        user_message="Can you forecast my solar production for the next 6 hours?",
        session={
            "user_id": 5,
            "username": "john_doe",
            "role": "user"
        }
    )
    
    # =========================================================================
    # Test Scenario 5: Admin User - Fleet Summary
    # =========================================================================
    test_scenario(
        scenario_name="Admin Gets Fleet Summary",
        user_message="Give me a summary of all installations",
        session={
            "user_id": 1,
            "username": "admin",
            "role": "admin"
        }
    )
    
    # =========================================================================
    # Test Scenario 6: Admin User - All Installations
    # =========================================================================
    test_scenario(
        scenario_name="Admin Lists All Installations",
        user_message="Show me all solar installations in the system",
        session={
            "user_id": 1,
            "username": "admin",
            "role": "admin"
        }
    )
    
    # =========================================================================
    # Test Scenario 7: User - Historical Data
    # =========================================================================
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    today = datetime.now().strftime("%Y-%m-%dT00:00:00Z")
    
    test_scenario(
        scenario_name="User Requests Historical Data",
        user_message=f"Show me my solar production data from {yesterday} to {today}",
        session={
            "user_id": 5,
            "username": "john_doe",
            "role": "user"
        }
    )
    
    # =========================================================================
    # Test Scenario 8: User - Access Denied Test
    # =========================================================================
    test_scenario(
        scenario_name="User Tries to Access Another's Installation",
        user_message="Show me installation ID 999 (that I don't own)",
        session={
            "user_id": 5,
            "username": "john_doe",
            "role": "user"
        }
    )
    
    # =========================================================================
    # Test Scenario 9: General Question
    # =========================================================================
    test_scenario(
        scenario_name="User Asks General Question",
        user_message="How does solar energy work?",
        session={
            "user_id": 5,
            "username": "john_doe",
            "role": "user"
        }
    )
    
    # =========================================================================
    # Test Scenario 10: Complex Multi-Step Query
    # =========================================================================
    test_scenario(
        scenario_name="User Complex Query",
        user_message="List my installations, then show me the current power for each one",
        session={
            "user_id": 5,
            "username": "john_doe",
            "role": "user"
        }
    )
    
    print_separator("Testing Complete")
    print("✅ All tests completed!")
    print(f"📊 View traces in LangSmith: https://smith.langchain.com/")
    print(f"📁 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}\n")


if __name__ == "__main__":
    main()
