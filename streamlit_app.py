"""
Streamlit entrypoint.

Run locally with:
  streamlit run streamlit_app.py
"""

def main() -> None:
    from frontend.app import main as frontend_main

    frontend_main()


if __name__ == "__main__":
    main()
