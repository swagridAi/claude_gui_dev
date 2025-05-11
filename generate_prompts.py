import csv

template = """
session{i}:
  claude_url: https://claude.ai/project/0196bd44-e4b5-75a9-bf40-0433b7b78507
  name: Code Tasks
  prompts:
  - With reference to Claude GUI Automation Refactoring Instructions assess the {file}. Briefly describe its purpose and how it fits into the overall project, the major functions or classes it contains. Do not refactor yet—just return your assessment.
  - Based on your assessment, propose a detailed refactoring plan for {file}. Include specific improvements such as function extraction, renaming, logic simplification, or restructuring; explain how these changes improve maintainability or readability; and note any dependencies or potential side effects. Wait for my approval before proceeding.
  - Apply the refactor to {file}. Output the complete refactored code in one block with minimal inline comments where necessary to explain any non-obvious changes. Do not modify other files or make assumptions about external dependencies unless instructed.
""".strip()

def generate_sessions_to_file(csv_path, output_path):
    with open(csv_path, newline='') as csvfile, open(output_path, 'w') as outfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader, start=1):
            file = row['file'].strip()
            output = template.format(i=i, file=file)
            outfile.write(output + "\n\n")  # Separate each block with double newline

# Run it
generate_sessions_to_file(r'C:\Users\User\python_code\claude_gui_prod\prompt_input.csv', 'sessions_output.txt')
