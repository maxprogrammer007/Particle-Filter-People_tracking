import os

# Define the structure of your LaTeX paper project
structure = {
    "tracking-paper": [
        "main.tex",
        "references.bib",
        "README.md",
        "sections/abstract.tex",
        "sections/introduction.tex",
        "sections/related_work.tex",
        "sections/methodology.tex",
        "sections/proposed_algorithm.tex",
        "sections/results_discussion.tex",
        "sections/conclusion.tex",
        "figures/pipeline_diagram.pdf",
        "figures/pareto_plot.png",
        "figures/sample_frame.png",
        "tables/comparison_table.tex",
        "style/IEEEtran.cls",
        "style/custom.sty"
    ]
}

# Create directories and placeholder files
for base, paths in structure.items():
    for path in paths:
        full_path = os.path.join(base, path)
        dir_name = os.path.dirname(full_path)
        os.makedirs(dir_name, exist_ok=True)

        # Create empty or templated files
        if full_path.endswith('.pdf') or full_path.endswith('.png'):
            continue  # Leave image placeholders
        with open(full_path, 'w') as f:
            if full_path.endswith('.tex'):
                f.write(f"% Section: {os.path.basename(full_path).replace('_', ' ').title()}\n")
            elif full_path.endswith('.bib'):
                f.write("% Bibliography entries go here\n")
            elif full_path.endswith('.md'):
                f.write("# Tracking Paper Project\n")
            elif full_path.endswith('.sty'):
                f.write("% Custom style definitions\n")
            elif full_path.endswith('.cls'):
                f.write("% Placeholder for IEEEtran.cls\n")

print("✅ Folder structure created under 'tracking-paper/'")
