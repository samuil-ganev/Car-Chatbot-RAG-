'''Tests the instantiation of the LLMConnector class using the provided credentials file.'''
import sys
from pathlib import Path
import pytest
from rag.llm_connector import LLMConnector


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def credentials_file() -> Path:  # Renamed from 'test_credentials_file' to 'credentials_file'
    '''Provides the path to the credentials.json file for use in LLMConnector tests.'''
    file_path = project_root / 'assets' / 'secrets' / 'credentials.json'
    return file_path

def test_llm_connector_instantiates(credentials_file: Path): # This now matches the fixture name
    ''' Verifies that LLMConnector initializes successfully with valid credentials.
    Fails the test if the credentials file is missing or if initialization fails.'''
    # The line below is now redundant because 'credentials_file' is passed by the fixture:
    # credentials_file = project_root / 'assets' / 'secrets' / 'credentials.json'

    if not credentials_file.exists():
        pytest.fail(f"Prerequisite failed: Credentials file not found at {credentials_file}")
    connector = LLMConnector(str(credentials_file))
    assert connector.llm is not None
