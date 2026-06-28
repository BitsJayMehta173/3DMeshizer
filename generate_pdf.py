import os
from markdown_pdf import MarkdownPdf
from markdown_pdf import Section

def generate_pdf():
    print("Generating MTech Thesis PDF...")
    
    pdf = MarkdownPdf(toc_level=2)
    
    with open("final_report.md", "r", encoding="utf-8") as f:
        md_content = f.read()

    # Add the markdown content as a section
    pdf.add_section(Section(md_content))
    
    # Save the PDF
    pdf.meta["title"] = "Final Project Report: 3D Mesh Compression"
    pdf.meta["author"] = "MTech Thesis"
    
    out_name = "Final_Project_Report.pdf"
    pdf.save(out_name)
    print(f"Successfully generated {out_name}")

if __name__ == "__main__":
    generate_pdf()
