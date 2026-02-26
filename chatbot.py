"""Deprecated Streamlit entrypoint.

This project now uses:
- `backend/app.py` (FastAPI backend)
- `frontend/` (TypeScript React UI)
"""


def main() -> None:
    print("Streamlit UI has been removed.")
    print("Start the backend with: uvicorn backend.app:app --reload")
    print("Start the frontend with: npm install && npm run dev (from ./frontend)")


if __name__ == "__main__":
    main()

