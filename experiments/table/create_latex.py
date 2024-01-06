import json



with open(os.path.join(script_folder, "save_stats"), "wb") as file:
    pickle.dump(dict(filtered_data), file)





# Parse the JSON data
header = json_data["header"]
data = json_data["data"]

# Generate LaTeX code for the table
latex_table = "\\begin{tabular}{|c|c|c|}\n"
latex_table += "\\hline\n"
latex_table += " & ".join(header) + " \\\\ \n"
latex_table += "\\hline\n"
for row in data:
    latex_table += " & ".join(str(row[column]) for column in header) + " \\\\ \n"
latex_table += "\\hline\n"
latex_table += "\\end{tabular}"

# Save to a .tex file
with open("table.tex", "w") as tex_file:
    tex_file.write(latex_table)