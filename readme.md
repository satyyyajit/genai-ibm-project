# 🎓 College Feedback Classifier

A Generative AI-based project that automatically classifies student feedback into categories such as Academics, Facilities, Administration, and Student Life using the FLAN-T5 base model and prompt engineering techniques.

---

## 📌 Project Overview

This project aims to simplify the analysis of open-ended student feedback collected by educational institutions. It uses few-shot prompting with a pre-trained FLAN-T5 model to predict the primary theme of each feedback entry. The output is saved in structured formats (CSV/JSON), enabling easy visualization and data-driven decisions.

---

## 🗂️ Dataset Info

The dataset is a synthetic collection of student feedback entries.  
It includes the following columns:

| Column Name      | Description                                 |
|------------------|---------------------------------------------|
| feedback_id      | Unique identifier for each feedback         |
| feedback_text    | Open-ended feedback provided by students    |
| category         | Labeled category (Academics, Facilities, etc.) |
| department       | Associated department                       |
| rating           | Rating given (1–5)                          |
| semester         | Semester when the feedback was given        |
| student_type     | Type of student (Undergraduate, Transfer, etc.) |

---

## 🎯 Objectives

- Classify open-ended student feedback into meaningful categories.
- Reduce manual effort in analyzing large volumes of qualitative feedback.
- Improve institutional response by surfacing common themes.
- Use prompt engineering with a language model to demonstrate few-shot classification.

---

## ⚙️ How It Works (Steps)

1. Load the dataset from `college_feedbacks.csv`.
2. Use prompt engineering with few-shot examples.
3. Tokenize the prompt using `AutoTokenizer`.
4. Use `google/flan-t5-base` to classify the feedback.
5. Append predictions to the dataset.
6. Save the output to both CSV and JSON.
7. Visualize prediction performance using bar charts.
8. Evaluate accuracy with scikit-learn's classification report.


## 🧰 Tools & Technologies Used

| Tool / Library        | Purpose                                      |
|-----------------------|----------------------------------------------|
| Python                | Main programming language                    |
| Pandas                | For reading, processing CSV data             |
| FLAN-T5 (Hugging Face)| Foundation model for few-shot classification |
| scikit-learn          | Evaluation (classification report)           |
| Matplotlib            | Visualizing predicted vs. actual categories  |
| CSV                   | Data input/output format                     |

---

## 🚀 How to Use This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/college-feedback-classifier.git
   cd college-feedback-classifier
   ```

2. Install required packages:
   *(optional if you want to download all of them earlier)*
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:
   - `college_feedback_classifier.ipynb`

4. Check outputs:
   - `tested_feedbacks.csv`
   - `json_tested_feedbacks.json`
   - Visualization plots inside the notebook

---

## 📁 Files in This Project

- `college_feedback_classifier.ipynb`: Main notebook with code
- `college_feedbacks.csv`: Input dataset
- `tested_feedbacks.csv`: CSV with predictions
- `json_tested_feedbacks.json`: JSON version of predictions
- `requirements.txt`: Python dependencies
- `README.md`: Project summary and documentation

---

