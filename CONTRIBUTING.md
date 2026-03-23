# Contributing to Hong Kong AQI Prediction

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues

- Search existing issues before creating a new one
- Use a clear, descriptive title
- Include steps to reproduce, expected behavior, and actual behavior
- Attach relevant logs or screenshots if applicable

### Pull Requests

1. **Fork the repository** and create a feature branch from `main`
2. **Keep your changes focused** - one feature or fix per PR
3. **Follow the existing code style** and structure
4. **Test your changes** before submitting
5. **Update documentation** if needed
6. **Reference issues** in your commit messages or PR description

### Development Workflow

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/hongkong_aqi_prediction.git
cd hongkong_aqi_prediction

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline (in order)
python src/01_data_collection.py
python src/02_preprocessing.py
python src/03_feature_engineering.py
python src/04_models.py
python src/09_final_evaluation.py
python src/10_report_visualization.py
```

### Code Quality Standards

- Write clear, readable code with appropriate comments
- Keep functions small and focused
- Handle errors explicitly
- Validate input at system boundaries
- Avoid hardcoded values (use constants or configuration)

### Notebook Guidelines

- Restart and run all cells before committing notebooks
- Clear outputs if they contain large data or sensitive information
- Export notebooks to HTML for preview: `jupyter nbconvert --to html notebooks/analysis.ipynb`

## Project Structure

```
hongkong_aqi_prediction/
|-- data/          # Data files (not committed to repo)
|-- models/        # Trained model files (.pkl)
|-- notebooks/     # Jupyter notebooks
|-- report/        # Generated reports and figures
|-- src/           # Python source code
`-- tmp/           # Temporary files
```

## Questions?

Feel free to open an issue for questions about the project or development process.
