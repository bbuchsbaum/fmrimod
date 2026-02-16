Contributing
============

We welcome contributions to fmrimod! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/YOUR_USERNAME/fmrimod.git
       cd fmrimod

3. Create a virtual environment:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install in development mode:

   .. code-block:: bash

       pip install -e ".[dev]"

Running Tests
-------------

We use pytest for testing:

.. code-block:: bash

    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=fmrimod

    # Run specific test file
    pytest tests/test_hrf_functions.py

Code Style
----------

We follow PEP 8 and use black for code formatting:

.. code-block:: bash

    # Format code
    black fmrimod tests

    # Check types
    mypy fmrimod

    # Run linting
    ruff check fmrimod

Making Changes
--------------

1. Create a new branch for your feature:

   .. code-block:: bash

       git checkout -b feature-name

2. Make your changes and add tests
3. Run the test suite to ensure everything passes
4. Commit your changes with a descriptive message
5. Push to your fork and create a pull request

Pull Request Guidelines
-----------------------

- Include tests for any new functionality
- Update documentation as needed
- Follow the existing code style
- Add a note to the changelog
- Ensure all tests pass

Areas for Contribution
----------------------

- Additional HRF types
- Performance optimizations
- Documentation improvements
- Example notebooks
- Bug fixes
- Cross-validation with R package

Questions?
----------

Feel free to open an issue on GitHub if you have questions or need help!