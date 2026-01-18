import os
import re
import sys
import importlib
import pkg_resources
from collections import defaultdict

def extract_imports_from_file(filepath):
    """Extract all import statements from a Python file."""
    imports = set()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Match: import module
        # Match: from module import ...
        # Match: from module.submodule import ...
        import_patterns = [
            r'^\s*import\s+([\w\.]+)',
            r'^\s*from\s+([\w\.]+)\s+import',
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            imports.update(matches)

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return imports

def get_base_package_name(module_name):
    """Get the base package name from a module path."""
    # Handle special cases
    special_mappings = {
        'langchain_community': 'langchain-community',
        'langchain_core': 'langchain-core',
        'langchain_text_splitters': 'langchain-text-splitters',
        'langchain_ollama': 'langchain-ollama',
        'langchain_chroma': 'langchain-chroma',
    }

    # Get base package name
    base = module_name.split('.')[0]

    # Check if it's a local module (starts with setup_ or is in current directory)
    if base.startswith('setup_') or base in ['indexer', 'test_vectordb', 'ask_questions', 'generate_requirements']:
        return None

    return special_mappings.get(base, base)

def get_package_version(package_name):
    """Get the installed version of a package."""
    try:
        version = pkg_resources.get_distribution(package_name).version
        return version
    except pkg_resources.DistributionNotFound:
        return None
    except Exception as e:
        return None

def scan_directory_for_imports(directory='.'):
    """Scan all Python files in directory for imports."""
    all_imports = set()

    print(f"Scanning directory: {directory}")
    print("="*60)

    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('generate_requirements'):
            filepath = os.path.join(directory, filename)
            print(f"Scanning: {filename}")
            imports = extract_imports_from_file(filepath)
            all_imports.update(imports)

    print(f"\nFound {len(all_imports)} unique imports")
    return all_imports

def filter_third_party_packages(imports):
    """Filter out built-in modules and get third-party packages."""
    third_party = set()
    builtin_modules = set(sys.builtin_module_names)
    stdlib_modules = {
        'os', 'sys', 're', 'json', 'logging', 'typing', 'collections',
        'datetime', 'io', 'pathlib', 'glob', 'itertools', 'functools'
    }

    for imp in imports:
        base_package = get_base_package_name(imp)

        # Skip if None (local module) or built-in/stdlib
        if base_package is None:
            continue

        if base_package not in builtin_modules and base_package not in stdlib_modules:
            third_party.add(base_package)

    # Add known dependencies that might not be directly imported
    additional_packages = {
        'chromadb',      # Used by Chroma vectorstore
        'ollama',        # Used by Ollama LLM
        'pypdf',         # Used by PyPDFLoader
        'flashrank',     # Used for reranking
        'rank_bm25',     # Used by BM25Retriever
    }

    third_party.update(additional_packages)

    return third_party

def generate_requirements():
    """Generate requirements.txt with exact versions."""

    print("\n" + "="*60)
    print("GENERATING REQUIREMENTS.TXT")
    print("="*60 + "\n")

    # Scan for imports
    all_imports = scan_directory_for_imports()

    # Filter third-party packages
    packages = filter_third_party_packages(all_imports)

    print("\n" + "="*60)
    print("DETECTED PACKAGES")
    print("="*60)

    # Get versions
    package_versions = {}
    for package in sorted(packages):
        version = get_package_version(package)
        if version:
            package_versions[package] = version
            print(f"✓ {package:30s} {version}")
        else:
            print(f"✗ {package:30s} NOT INSTALLED")

    # Write requirements.txt
    requirements_file = 'requirements.txt'

    print("\n" + "="*60)
    print(f"WRITING {requirements_file}")
    print("="*60)

    with open(requirements_file, 'w') as f:
        f.write("# Requirements for PDF Chatbot RAG System\n")
        f.write("# Generated automatically\n\n")

        for package in sorted(package_versions.keys()):
            line = f"{package}=={package_versions[package]}\n"
            f.write(line)
            print(f"Added: {line.strip()}")

    print(f"\n✓ Requirements file created: {requirements_file}")
    print(f"✓ Total packages: {len(package_versions)}")

    # Print installation command
    print("\n" + "="*60)
    print("INSTALLATION COMMAND")
    print("="*60)
    print(f"\nTo install all dependencies, run:")
    print(f"pip install -r {requirements_file}")

    # Show missing packages
    missing = packages - set(package_versions.keys())
    if missing:
        print("\n" + "="*60)
        print("WARNING: MISSING PACKAGES")
        print("="*60)
        print("\nThe following packages are imported but not installed:")
        for pkg in sorted(missing):
            print(f"  - {pkg}")
        print("\nYou may need to install these manually or they might be sub-packages.")

if __name__ == "__main__":
    try:
        generate_requirements()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
