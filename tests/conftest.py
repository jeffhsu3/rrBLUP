# tests/conftest.py
import pytest

# Global state to avoid multiple import attempts / rpy2 activations
R_SETUP_DONE = False
R_RRBLUP_AVAILABLE = False
R_RRBLUP_IMPORT_ERROR = ""
RRBLUP_R_PACKAGE = None

def pytest_configure(config):
    """
    Performs R and rrBLUP package availability check once when pytest starts.
    Activates rpy2 interfaces and imports the rrBLUP R package.
    Stores the state globally to be used by fixtures.
    """
    global R_SETUP_DONE, R_RRBLUP_AVAILABLE, R_RRBLUP_IMPORT_ERROR, RRBLUP_R_PACKAGE
    if not R_SETUP_DONE:
        try:
            from rpy2.robjects.packages import importr
            import rpy2.robjects.numpy2ri
            import rpy2.robjects.pandas2ri

            # Attempt to activate interfaces
            rpy2.robjects.numpy2ri.activate()
            rpy2.robjects.pandas2ri.activate()
            
            # Attempt to import rrBLUP
            RRBLUP_R_PACKAGE = importr('rrBLUP')
            R_RRBLUP_AVAILABLE = True
        except Exception as e:
            R_RRBLUP_IMPORT_ERROR = str(e)
            # R_RRBLUP_AVAILABLE remains False
        
        R_SETUP_DONE = True

@pytest.fixture(scope="session")
def rrblup_r_package():
    """
    Pytest fixture that provides the imported R rrBLUP package.
    Skips the test if R or rrBLUP is not available.
    """
    if not R_RRBLUP_AVAILABLE:
        pytest.skip(f"Skipping R dependent test: rrBLUP R package not available or rpy2 setup issue. Error: {R_RRBLUP_IMPORT_ERROR}")
    return RRBLUP_R_PACKAGE
