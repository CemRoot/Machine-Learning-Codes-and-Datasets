# Contributing to Machine Learning Codes and Datasets

First off, thank you for considering contributing to this project! It's people like you that make this repository a great learning resource for the machine learning community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. By participating, you are expected to uphold this commitment.

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## 🤝 How Can I Contribute?

### 1. Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Screenshots** (if applicable)
- **Environment details** (OS, Python version, etc.)

**Example:**
```markdown
**Bug:** K-Means clustering fails with empty dataset

**Steps to Reproduce:**
1. Load empty CSV file
2. Run k_means_clustering.py
3. Observe error

**Expected:** Graceful error message
**Actual:** Unhandled exception

**Environment:** Python 3.9, Windows 10
```

### 2. Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear use case** for the enhancement
- **Detailed description** of the proposed functionality
- **Potential implementation** approach (if you have ideas)
- **Examples** from other projects (if applicable)

### 3. Adding New Algorithms

Want to add a new ML algorithm? Great! Please ensure:

- The algorithm is well-documented
- Code follows our style guidelines
- Includes a sample dataset
- Has both `.py` and `.ipynb` versions
- Includes visualization where applicable

### 4. Improving Documentation

Documentation improvements are always welcome:

- Fix typos or grammatical errors
- Clarify confusing explanations
- Add more examples
- Translate documentation to other languages

### 5. Adding Datasets

When contributing datasets:

- Ensure you have the right to share the data
- Include a data dictionary/README
- Use standard formats (CSV, JSON, etc.)
- Keep file sizes reasonable (<10MB preferred)
- Include source attribution

## 🚀 Getting Started

### 1. Fork the Repository

Click the "Fork" button at the top right of the repository page.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/Machine-Learning-Codes-and-Datasets.git
cd Machine-Learning-Codes-and-Datasets
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-svm-classifier`
- `fix/kmeans-empty-data-bug`
- `docs/improve-readme`
- `dataset/add-iris-data`

## 💻 Development Process

### 1. Make Your Changes

- Write clean, readable code
- Add comments for complex logic
- Follow existing code structure
- Test your changes thoroughly

### 2. Test Your Code

```bash
# Run tests (if applicable)
pytest

# Check code formatting
black --check .

# Check code style
flake8 .
```

### 3. Update Documentation

- Update README.md if adding new features
- Add docstrings to functions
- Include inline comments for complex code
- Update CHANGELOG.md (if applicable)

## 🎨 Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Good Example
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_model(X_train, y_train):
    """
    Train a machine learning model.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels

    Returns:
    --------
    model : object
        Trained model
    """
    model = SomeMLModel()
    model.fit(X_train, y_train)
    return model
```

### Key Points

- **Indentation:** 4 spaces (no tabs)
- **Line length:** Max 88 characters (Black default)
- **Imports:** Standard library → Third-party → Local
- **Naming:**
  - Variables: `snake_case`
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Jupyter Notebook Guidelines

- Clear, descriptive markdown cells
- One concept per cell
- Include visualizations
- Clean output (run all cells before committing)
- Add cell numbers for reference

### R Code Style (if applicable)

```r
# Good Example
trainModel <- function(xTrain, yTrain) {
  # Train the model
  model <- lm(yTrain ~ xTrain)
  return(model)
}
```

## 📝 Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(regression): add polynomial regression implementation

- Implement polynomial features
- Add visualization
- Include sample dataset

Closes #123
```

```bash
fix(kmeans): handle empty dataset gracefully

- Add input validation
- Raise informative error message
- Add unit tests

Fixes #456
```

```bash
docs(readme): update installation instructions

- Add conda installation steps
- Clarify Python version requirements
```

## 🔄 Pull Request Process

### 1. Update Your Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

- Go to the original repository
- Click "New Pull Request"
- Select your branch
- Fill out the PR template

### PR Title Format

```
[Type] Brief description
```

Examples:
- `[Feature] Add Decision Tree Classifier`
- `[Fix] Resolve data preprocessing bug`
- `[Docs] Improve README structure`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
Describe how you tested your changes

## Screenshots (if applicable)
Add screenshots here

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed the code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No new warnings
- [ ] Added tests (if applicable)
- [ ] All tests pass
```

### 4. Code Review

- Be responsive to feedback
- Make requested changes promptly
- Be open to suggestions
- Ask questions if unclear

### 5. After Approval

Once approved and merged:
- Delete your branch (optional)
- Pull the latest changes
- Celebrate! 🎉

## ❓ Questions?

If you have questions:

1. Check [existing issues](https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets/issues)
2. Check [discussions](https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets/discussions)
3. Create a new discussion or issue

## 🙏 Recognition

All contributors will be recognized in our README. Thank you for making this project better!

---

## 🇹🇷 Türkçe Katkı Rehberi

### Nasıl Katkıda Bulunabilirsiniz?

1. **Hata Bildirme**: Karşılaştığınız hataları detaylı şekilde bildirin
2. **Özellik Önerme**: Yeni algoritma veya özellik önerileri sunun
3. **Kod Geliştirme**: Yeni algoritmalar ekleyin veya mevcut kodu iyileştirin
4. **Dokümantasyon**: Türkçe dokümantasyon ekleyin veya iyileştirin
5. **Veri Seti**: Yeni veri setleri ekleyin

### Başlangıç Adımları

```bash
# Depoyu fork edin ve klonlayın
git clone https://github.com/KULLANICI_ADINIZ/Machine-Learning-Codes-and-Datasets.git

# Sanal ortam oluşturun
python -m venv ml_env
source ml_env/bin/activate  # Windows: ml_env\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Yeni bir branch oluşturun
git checkout -b feature/yeni-ozellik

# Değişikliklerinizi yapın ve commit edin
git commit -m "feat: yeni özellik eklendi"

# Push yapın ve PR oluşturun
git push origin feature/yeni-ozellik
```

### İletişim

Sorularınız için:
- Issue açın
- Discussion başlatın
- Detaylı açıklama yapın

---

<div align="center">

**Thank you for contributing! 🚀**

**Katkılarınız için teşekkürler! 🚀**

</div>
