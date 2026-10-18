"""
Launcher for Streamlit Power Showcase
Runs the dashboard file located in 'p4  Web Based Dashboard/main.py'.
"""
import runpy
from pathlib import Path

dashboard_path = Path(__file__).parent / "p4  Web Based Dashboard" / "main.py"

if __name__ == "__main__":
    runpy.run_path(str(dashboard_path), run_name="__main__")
