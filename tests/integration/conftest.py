"""
Shared fixtures for integration tests.

Isolates every test run to a throw-away SQLite file so no test data
ever touches the real database at /tmp/stylevid2/app.db.
"""
import os
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="session", autouse=True)
def _isolated_db():
    """
    Redirect the app to a temp SQLite DB for the entire test session.

    Patches both the module-level engine/SessionLocal (used by init_db and
    Celery workers) and the FastAPI get_db dependency (used by API routes),
    so no test query ever touches the real database.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    test_url = f"sqlite:///{tmp.name}"

    import backend.db.database as db_module
    from backend.db.models import Base
    from backend.db.database import get_db
    from backend.api.main import app

    test_engine = create_engine(test_url, connect_args={"check_same_thread": False})
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Create all tables (including the migration columns) on the test DB
    Base.metadata.create_all(bind=test_engine)

    # Patch module-level globals so init_db() and SessionLocal() hit the test DB
    _orig_engine = db_module.engine
    _orig_session = db_module.SessionLocal
    db_module.engine = test_engine
    db_module.SessionLocal = TestSession

    # Override FastAPI's get_db dependency for all routes
    def _override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db

    yield

    # Teardown: restore real DB, delete temp file
    app.dependency_overrides.clear()
    db_module.engine = _orig_engine
    db_module.SessionLocal = _orig_session
    test_engine.dispose()
    os.unlink(tmp.name)
