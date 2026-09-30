# Research Paper Insight Generator

## 

## 

## \## 1. Project Overview

## 

## The Research Paper Insight Generator is an automated pipeline that extracts useful information from research paper PDFs and converts it into a structured CSV dataset.

## 

## The project reduces the repetitive manual work involved in reading multiple research papers and collecting their basic information.

## 

## \## 2. Problem Statement

## 

## Research papers contain useful information such as titles, authors, abstracts, keywords and other textual content. Extracting and organising this information manually from multiple PDFs can be repetitive and time-consuming.

## 

## This project automates the extraction and processing of research papers into a structured dataset.

## 

## \## 3. Proposed Solution

## 

## The system takes research paper PDFs as input and processes them through a Python-based pipeline.

## 

## \### Pipeline

## 

## PDF Research Papers

## &#x20;       ↓

## Text Extraction

## &#x20;       ↓

## Text Cleaning

## &#x20;       ↓

## Metadata Extraction

## &#x20;       ↓

## Abstract Extraction

## &#x20;       ↓

## Extractive Summarisation

## &#x20;       ↓

## TF-IDF Keyword Extraction

## &#x20;       ↓

## Data Quality Checks

## &#x20;       ↓

## Structured CSV Dataset

## 

## \## 4. Features

## 

## The system extracts and generates:

## 

## \- Paper ID

## \- File name

## \- Title

## \- Authors

## \- Preprint date

## \- Publication year

## \- Abstract

## \- Keywords

## \- Number of pages

## \- Word count

## \- Extractive summary

## \- TF-IDF keywords

## 

## The pipeline also performs basic data quality checks for:

## 

## \- Duplicate paper IDs

## \- Empty abstracts

## \- Empty summaries

## 

## \## 5. Technologies Used

## 

## \- Python

## \- PyMuPDF

## \- Pandas

## \- Scikit-learn

## \- Git

## \- GitHub

## \- Linux/open-source environment through Docker

## \- Docker

## \- Docker Compose

## 

## \## 6. Project Structure

## 

## ```text

## research-paper-insight-generator/

## │

## ├── app/

## │   └── main.py

## │

## ├── input/

## │   └── Research paper PDFs

## │

## ├── output/

## │   └── research\_paper\_insights.csv

## │

## ├── Research\_Paper\_Insight\_Generator\_FINAL.ipynb

## ├── requirements.txt

## ├── Dockerfile

## ├── docker-compose.yml

## ├── .dockerignore

## ├── .gitignore

## ├── README.md

## └── LICENSE

