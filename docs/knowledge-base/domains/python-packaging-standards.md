# Python Packaging Standards and Specifications

## Overview
This document catalogs the authoritative standards, specifications, and conventions that govern Python package management and dependency declaration in the Ordinis project.

## Formal Standards Bodies

### IEEE Standards
**IEEE** (Institute of Electrical and Electronics Engineers)
- **IEEE 829**: Software test documentation
- **Coverage**: Software testing, NOT package management

### ISO/IEC Standards
**ISO/IEC 12207**: Systems and software engineering — Software life cycle processes
- **Coverage**: High-level software lifecycle processes
- **Relevance**: Process framework, not package formats

**ISO/IEC 25010**: Systems and software Quality Requirements and Evaluation (SQuaRE)
- **Coverage**: Software quality models
- **Relevance**: Quality characteristics, not dependency management

**Note**: No IEEE, ISO, or IEC standards currently govern Python package manager file formats or dependency specifications.

## Python Enhancement Proposals (PEPs)

Python's internal standards process for language and tooling improvements.

### PEP 440: Version Identification and Dependency Specification
- **Status**: Final
- **URL**: https://peps.python.org/pep-0440/
- **Scope**: Defines version numbering scheme and comparison
- **Key Elements**:
  - Version format: `N.N[.N]*[{a|b|rc}N][.postN][.devN]`
  - Version specifiers: `==`, `!=`, `<=`, `>=`, `<`, `>`, `~=`, `===`
  - Examples: `>=1.0.0`, `~=2.1.0` (compatible release)

### PEP 508: Dependency Specification for Python Software Packages
- **Status**: Final
- **URL**: https://peps.python.org/pep-0508/
- **Scope**: Standard format for declaring dependencies
- **Syntax**:
  ```
  name [extras] [version-spec] [; environment-marker]
  ```
- **Examples**:
  ```
  requests>=2.0.0
  numpy>=1.20.0; python_version>='3.8'
  pytest>=7.0; extra=='dev'
  ```

### PEP 621: Storing project metadata in pyproject.toml
- **Status**: Final
- **URL**: https://peps.python.org/pep-0621/
- **Scope**: Standard for declaring project metadata and dependencies in `pyproject.toml`
- **Key Sections**:
  - `[project]`: Core metadata
  - `[project.dependencies]`: Runtime dependencies
  - `[project.optional-dependencies]`: Optional dependency groups
- **Supersedes**: `setup.py`, `setup.cfg` for metadata declaration

### PEP 517: A build-system independent format for source trees
- **Status**: Final
- **URL**: https://peps.python.org/pep-0517/
- **Scope**: Defines `[build-system]` table in `pyproject.toml`
- **Purpose**: Specifies build backend (setuptools, flit, hatch, etc.)

### PEP 518: Specifying Minimum Build System Requirements
- **Status**: Final
- **URL**: https://peps.python.org/pep-0518/
- **Scope**: Introduced `pyproject.toml` as configuration file
- **Purpose**: Declare build dependencies before installation

## Python Packaging Authority (PyPA)

Community-driven guidance for Python packaging.

### PyPA Specifications
- **URL**: https://packaging.python.org/en/latest/specifications/
- **Authority**: Python Packaging Authority
- **Status**: Authoritative community guidance (not formal standard)

### Key PyPA Guides
1. **Packaging Python Projects**: https://packaging.python.org/en/latest/tutorials/packaging-projects/
2. **Dependency Management**: https://packaging.python.org/en/latest/tutorials/managing-dependencies/
3. **Tool Recommendations**: https://packaging.python.org/en/latest/guides/tool-recommendations/

## Conda Specifications

### Conda Environment File Format
- **Documentation**: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- **File**: `environment.yml`
- **Authority**: Anaconda, Inc.
- **Status**: Tool-specific format (not cross-tool standard)
- **Key Sections**:
  ```yaml
  name: env-name
  channels:
    - conda-forge
    - defaults
  dependencies:
    - python=3.11
    - numpy>=1.20
    - pip:
      - requests>=2.0
  ```

### Conda Package Specification
- **Version Matching**: Uses "match spec" format
  - `package`: any version
  - `package=1.2`: version 1.2.*
  - `package=1.2.3`: exact version
  - `package>=1.2`: minimum version
  - `package=1.2.*`: glob pattern

### Conda-Lock
- **Tool**: https://github.com/conda/conda-lock
- **Purpose**: Generate reproducible lock files from `environment.yml`
- **Output**: Platform-specific lock files with exact pins
- **Status**: Community tool, not official standard

## Ordinis Project Conventions

### Primary Dependency Declaration
**File**: `pyproject.toml`
- **Authority**: PEP 621
- **Use**: All Python package dependencies
- **Format**: TOML with `[project.dependencies]` and `[project.optional-dependencies]`

### Conda Environment (GPU/System Dependencies)
**File**: `environment.yml`
- **Authority**: Conda specification
- **Use**: CUDA, PyTorch, system-level packages
- **Format**: YAML with flexible version constraints (`>=`)

### Lock Files (Reproducibility)
**Files**:
- `requirements.txt` (via `pip freeze`)
- `environment-lock.yml` (via `conda env export`)

**Use**: CI/CD, production deployments requiring exact reproducibility

### Version Constraint Philosophy
- **Development**: Use minimum version constraints (`>=`)
- **Testing/CI**: Lock to exact versions
- **Documentation**: Specify minimum tested versions
- **Production**: Use lock files for deployments

## References

### Official Documentation
1. **Python Packaging User Guide**: https://packaging.python.org
2. **PEP Index**: https://peps.python.org
3. **Conda Documentation**: https://docs.conda.io
4. **PyPI**: https://pypi.org
5. **pip Documentation**: https://pip.pypa.io

### Version Control
- **Last Updated**: 2025-12-15
- **Reviewed By**: Development Team
- **Next Review**: Quarterly (or when major PEP changes occur)

## Compliance Checklist

- [x] `pyproject.toml` follows PEP 621
- [x] Dependencies use PEP 440 version specifiers
- [x] `environment.yml` uses flexible constraints
- [ ] Lock files generated for production
- [ ] Dependencies reviewed quarterly
- [ ] Security audits on dependencies (via `pip-audit` or `safety`)

## Notes

**Standards Gap**: There are no IEEE/IEC/ISO standards governing Python package management file formats. The ecosystem relies on:
- **PEPs**: Python's internal standards process
- **PyPA**: Community-driven guidance
- **Tool Maintainers**: Conda (Anaconda), pip (PyPA), poetry (Python Poetry)

This is intentional—the Python community prefers agile, community-driven standards over formal standardization processes.
