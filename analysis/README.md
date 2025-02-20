# Comparing Reasoning Framework Similarity Before and After Editing

This folder provides tools to compare the similarity of reasoning frameworks before and after editing.

## Steps to Use

### 1. Process Reasoning Results
Ensure that the reasoning results are processed into the following format:
- A list where:
  - Each element corresponds to the reasoning path for a question.
  - Each reasoning path is a list of steps.

### 2. Extract Reasoning Frameworks
Run the following command to extract the reasoning frameworks:

```bash
python get_relations.py
```

Make sure the lists for the edited and unedited versions are aligned one-to-one.

### 3. Compute Similarity
Run the following command to calculate the similarity between the reasoning frameworks:

```bash
python get_sim.py
```

This will output the similarity score between the reasoning frameworks before and after the edits.

## Repository Structure
```
.
├── get_relations.py  # Script to extract reasoning frameworks
├── get_sim.py        # Script to compute reasoning similarity
├