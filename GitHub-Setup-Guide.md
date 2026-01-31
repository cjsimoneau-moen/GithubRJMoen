# GitHub Portfolio Setup Guide for Riley Moen

This guide will help you create an impressive GitHub portfolio that showcases your data science and machine learning skills to potential employers.

## 🎯 Portfolio Goals

1. **Demonstrate Technical Skills:** Show proficiency in Python, ML, data analysis
2. **Show Project Complexity:** Projects that go beyond tutorial-level work
3. **Prove You Can Finish:** Complete projects with documentation and results
4. **SEO for Recruiters:** Optimize for GitHub search and Google
5. **Easy to Navigate:** Clear structure that recruiters can understand quickly

---

## 📋 Quick Start Checklist

### Week 1: Setup & First Project
- [ ] Create GitHub account (if you don't have one)
- [ ] Set up profile README (use Portfolio-README.md provided)
- [ ] Upload CNN image classification project
- [ ] Add professional profile picture
- [ ] Pin your best repositories

### Week 2: Add More Projects
- [ ] Upload 2-3 more projects from your Master's work
- [ ] Create detailed README for each project
- [ ] Add visualizations and results
- [ ] Ensure all code is well-commented

### Week 3: Polish & Optimize
- [ ] Add GitHub badges to READMEs
- [ ] Create requirements.txt for each project
- [ ] Add LICENSE files
- [ ] Write project descriptions
- [ ] Test all code runs without errors

### Week 4: Promote
- [ ] Share on LinkedIn
- [ ] Add to resume and portfolio site
- [ ] Star relevant repositories in your field
- [ ] Contribute to open source (optional)

---

## 🏗️ Repository Structure

### Option 1: Monorepo (Recommended for Portfolio)
One main repository containing all projects:

```
riley-moen-portfolio/
├── README.md (main portfolio page)
├── 01-image-classification-cnn/
│   ├── README.md
│   ├── src/
│   ├── notebooks/
│   └── results/
├── 02-llm-text-classification/
│   ├── README.md
│   ├── src/
│   └── results/
├── 03-numerical-linear-algebra/
│   ├── README.md
│   └── matlab/
└── 04-time-series-lstm/
    ├── README.md
    └── src/
```

**Pros:**
- Easy to navigate
- One link to share
- Consistent structure
- Better for showcasing

### Option 2: Individual Repos
Separate repository for each project:

```
yourusername/cnn-image-classification
yourusername/llm-text-classifier
yourusername/numerical-methods-toolkit
```

**Pros:**
- More professional looking
- Can pin top 6 repos
- Better for large projects
- Easier to share specific projects

**Recommendation:** Start with Option 1 (monorepo), then later split into individual repos if projects grow large.

---

## 📝 Project Priority List

### Must-Have Projects (Do These First):

1. **CNN Image Classification** ✅ (Provided)
   - Shows: Deep learning, PyTorch, computer vision
   - Time: Already written - just needs your data
   - Impact: HIGH - demonstrates core ML skills

2. **Numerical Linear Algebra Toolkit**
   - Shows: MATLAB, Python, mathematical rigor
   - Time: ~4-6 hours to document existing work
   - Impact: MEDIUM-HIGH - showcases math background

3. **Statistical Analysis Dashboard**
   - Shows: Data visualization, Streamlit, communication
   - Time: ~6-8 hours
   - Impact: HIGH - shows you can build user-facing tools

### Nice-to-Have Projects:

4. **LLM Fine-tuning Project**
   - Shows: NLP, transformers, Hugging Face
   - Time: ~8-10 hours
   - Impact: VERY HIGH - LLMs are hot right now

5. **Time Series Forecasting**
   - Shows: LSTM, sequential data, finance applications
   - Time: ~6-8 hours
   - Impact: MEDIUM - relevant for fintech companies

6. **A/B Testing Framework**
   - Shows: Statistics, experiment design, business impact
   - Time: ~4-6 hours
   - Impact: MEDIUM-HIGH - shows business acumen

---

## 🎨 Making Your README Stand Out

### Essential Elements:

1. **Clear Title & Badges**
```markdown
# Project Name

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
```

2. **Quick Summary (First 3 Lines)**
```markdown
**One-sentence description of what the project does**

Key Achievement: Specific metric or result

Tech Stack: List of main technologies
```

3. **Visual Results EARLY**
   - Put your best visualization in the first screen
   - Confusion matrix, training curves, demo GIF
   - Recruiters skim - make it visual!

4. **Project Structure**
   - Show file organization
   - Helps recruiters navigate
   - Shows professional thinking

5. **Results Section with Numbers**
   - "Achieved 94% accuracy"
   - "Reduced training time by 30%"
   - "Processed 50,000 images"

6. **What You Learned**
   - Shows reflection
   - Demonstrates growth mindset
   - Humanizes the project

### DON'T Include:
- ❌ Lengthy installation instructions (keep brief)
- ❌ Every single detail (save for actual code)
- ❌ Apologies for code quality
- ❌ "This is a class project" (just present it professionally)

---

## 💡 Converting Your Master's Work to Projects

You have tons of material from your Master's program. Here's how to convert it:

### From: Class Assignment
**To: Professional Project**

**Before:**
> "MATH 574 Homework 3 - Implemented LU decomposition"

**After:**
> "# Numerical Linear Algebra Toolkit
> 
> Efficient matrix factorization algorithms with complexity analysis
> 
> - Implemented LU, QR, and SVD decompositions
> - Benchmarked against NumPy (achieved within 5% performance)
> - Analyzed numerical stability across different condition numbers"

### The Transformation Process:

1. **Remove Class Context**
   - Don't mention it's homework
   - Frame as independent research

2. **Add Context**
   - Why this problem matters
   - Real-world applications
   - What you learned

3. **Professionalize**
   - Clean up code
   - Add docstrings
   - Create visualizations

4. **Show Results**
   - Performance metrics
   - Comparison plots
   - Key findings

---

## 🔍 SEO for Recruiters

Recruiters search GitHub for:
- "python machine learning"
- "pytorch cnn"
- "data science denver"

**Optimize your repositories:**

1. **Repository Names**
   - ✅ `cnn-image-classification-pytorch`
   - ❌ `project1` or `homework3`

2. **Repository Description**
   - ✅ "CNN for image classification using PyTorch, achieving 94% accuracy on CIFAR-10"
   - ❌ "My ML project"

3. **Topics/Tags**
   - Add topics: `machine-learning`, `pytorch`, `computer-vision`, `deep-learning`, `python`

4. **README Keywords**
   - Include: Python, PyTorch, TensorFlow, ML, data science
   - Natural placement in headings and text

---

## 🚀 GitHub Profile README

Create a special README that appears on your profile:

1. Create a repository named `yourusername` (same as your GitHub username)
2. Add a README.md to that repository
3. Use the Portfolio-README.md I provided as a template

**This README appears on your profile page and is the FIRST thing recruiters see.**

---

## 📊 GitHub Profile Optimization

### Profile Settings:

1. **Profile Picture**
   - Professional headshot
   - Same as LinkedIn
   - Clear, high quality

2. **Bio**
   - "MS Applied Math @ Colorado School of Mines | ML Engineer | Python, PyTorch, TensorFlow | Seeking DS roles in Denver"

3. **Location**
   - "Denver, CO" (helps local recruiters find you)

4. **Website**
   - Link to your portfolio site

5. **Company**
   - "Colorado School of Mines" (or leave blank)

### Pinned Repositories:

Pin your 6 best repositories:
1. Main portfolio repository
2. CNN project
3. LLM/NLP project
4. Numerical methods
5. Dashboard/visualization
6. Statistical analysis

**Order matters!** Put your strongest work first.

---

## 📈 Activity & Engagement

### Green Squares Matter

Recruiters look at your contribution graph. Show consistent activity:

1. **Commit Regularly**
   - Even small commits (fix typo, update README)
   - Aim for 3-4 commits per week
   - Batch your work but spread commits

2. **Star Relevant Repos**
   - PyTorch, TensorFlow, scikit-learn
   - Shows you follow the field
   - ~50-100 starred repos looks good

3. **Follow People**
   - ML researchers
   - Denver data scientists
   - Colleagues from Mines

### Don't:
- ❌ Fake activity with empty commits
- ❌ Commit sensitive information
- ❌ Have massive gaps (looks inactive)

---

## 🎯 Quick Wins (Do This Weekend)

### Saturday (4 hours):
1. Create GitHub account / clean up existing
2. Upload main portfolio README
3. Create one complete project repository
4. Add profile picture and bio

### Sunday (4 hours):
1. Upload 2 more projects from your coursework
2. Write basic READMEs for each
3. Add requirements.txt files
4. Pin your repositories

**Total: 8 hours to professional portfolio**

---

## 🔧 Technical Setup

### Git Basics:

```bash
# First time setup
git config --global user.name "Riley Moen"
git config --global user.email "your.email@example.com"

# Create and push a repository
cd your-project-folder
git init
git add .
git commit -m "Initial commit: Add CNN project"
git branch -M main
git remote add origin https://github.com/yourusername/repo-name.git
git push -u origin main
```

### .gitignore for Data Science:

```
# Python
__pycache__/
*.pyc
*.pyo
.Python
venv/
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
data/
*.csv
*.h5
*.pkl

# Models
*.pth
*.pt
models/

# IDE
.vscode/
.idea/
*.swp
```

---

## 📧 Repository Templates

I've created the following files for you:

1. **Portfolio-README.md** - Main profile README
2. **Project1-CNN-README.md** - Example project README
3. **model.py** - Example well-documented code

Use these as templates for your own projects!

---

## ✅ Quality Checklist

Before pushing ANY repository, verify:

### Code Quality:
- [ ] Code runs without errors
- [ ] Functions have docstrings
- [ ] Variables have meaningful names
- [ ] No hardcoded paths or secrets
- [ ] requirements.txt is complete

### Documentation:
- [ ] README has clear title and description
- [ ] Installation instructions work
- [ ] Usage examples provided
- [ ] Results/metrics included
- [ ] License file added (MIT is fine)

### Professional:
- [ ] No "TODO" or "FIX THIS" comments
- [ ] No profanity or jokes in code/comments
- [ ] No class-specific references
- [ ] Proper grammar in README

---

## 🎓 Example Workflow

### Converting Your CNN Class Project:

1. **Gather Materials** (15 min)
   - Find your code files
   - Locate any results/plots
   - Check what datasets you used

2. **Clean Code** (1 hour)
   - Remove class-specific comments
   - Add docstrings
   - Rename variables if needed
   - Ensure it runs

3. **Create README** (45 min)
   - Use provided template
   - Add your results
   - Create architecture diagram
   - Write what you learned

4. **Add Visualizations** (30 min)
   - Training curves
   - Confusion matrix
   - Sample predictions
   - Save as PNGs in results/

5. **Push to GitHub** (15 min)
   - Initialize git
   - Add files
   - Commit and push
   - Add topics/description

**Total: ~3 hours for a complete professional project**

---

## 🌟 Standing Out

### What Makes a Portfolio EXCELLENT:

1. **Live Demos**
   - Streamlit app deployed
   - Interactive visualizations
   - Anyone can try it

2. **Comprehensive Documentation**
   - Not just "what" but "why"
   - Design decisions explained
   - Lessons learned

3. **Production Thinking**
   - Error handling
   - Testing
   - Docker containers
   - CI/CD (advanced)

4. **Unique Angles**
   - Apply ML to unusual domains
   - Your math background + ML
   - Colorado-specific projects

### Project Ideas Using YOUR Background:

1. **"Mathematical Analysis of Neural Network Convergence"**
   - Combines your math + ML
   - Shows theoretical depth
   - Very differentiating

2. **"Numerical Methods for Large-Scale ML"**
   - Optimization algorithms
   - Convergence analysis
   - Shows unique skillset

3. **"Colorado Weather Prediction Using LSTMs"**
   - Local relevance (Denver!)
   - Time series + ML
   - Can show at interviews

---

## 📞 Next Steps

1. **This Weekend:** Set up basic portfolio (use provided files)
2. **Next Week:** Upload 3 projects from Master's work
3. **Week 3:** Polish and add visualizations
4. **Week 4:** Share on LinkedIn and add to applications

---

## 💬 Need Help?

Common issues and solutions:

**Q: "My code is messy from class, should I upload it?"**
A: Clean it up first. 3 hours of cleanup is worth it.

**Q: "I don't have results/plots saved"**
A: Re-run your code and generate them. Add to README.

**Q: "Should I include failed experiments?"**
A: NO. Only include successful, complete work.

**Q: "How many projects do I need?"**
A: Minimum 3, ideal is 5-6. Quality > quantity.

---

## 🎯 Remember

Your GitHub is:
- Your technical resume
- Your portfolio
- Your first impression

Invest 10-15 hours now to create a professional portfolio that will work for you 24/7 in your job search.

**Most important:** Start today. Even one good project is infinitely better than zero.

---

**Files Provided:**
1. Portfolio-README.md (main profile)
2. Project1-CNN-README.md (example project)
3. model.py (example code)

Copy these, customize with your info, and push to GitHub this weekend!

Good luck! 🚀
