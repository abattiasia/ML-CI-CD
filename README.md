# Using GitHub Actions for Testing Machine Learning Models

## Introduction

In the rapidly evolving field of machine learning, maintaining high-quality models is essential. Continuous integration and continuous deployment (CI/CD) practices, such as those enabled by GitHub Actions, can streamline the process of testing, evaluating, and deploying machine learning models. This report outlines a GitHub Actions workflow for model evaluation, detailing the steps involved in automating the testing of a machine learning model each time changes are made to the codebase.

## Workflow Overview

The GitHub Actions workflow defined in this report is triggered on pull requests to the `main` branch. The primary objective of this workflow is to evaluate the performance of a machine learning model whenever changes are made, ensuring that only models that meet the required performance criteria are merged and potentially deployed.

### Workflow Definition

The workflow is defined in a YAML file, typically located in the `.github/workflows` directory of the repository. Below is the code for the workflow:

```yaml
name: Model Evaluation

# Triggers: events 
on:
    pull_request:
        branches:
            - main

# Code 
jobs:
    model-evaluation:
        runs-on: ubuntu-latest  # OS: Linux 

        # Steps 
        steps:
            - name: Checkout code 
              uses: actions/checkout@v4

            - name: Setup Python 3.11
              uses: actions/setup-python@v4
              with:
                python-version: 3.11

            - name: Setup requirements 
              run: |
                python -m pip install --upgrade pip
                pip install -r requirements.txt

            - name: Run Model Evaluation
              run: python app.py 

            - name: Show metrics 
              uses: actions/upload-artifact@v3
              with:
                name: Evaluation Results
                path: metrics.txt
 
 
