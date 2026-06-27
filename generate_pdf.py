import os
from markdown_pdf import MarkdownPdf
from markdown_pdf import Section

def generate_pdf():
    print("Generating MTech Thesis PDF...")
    
    pdf = MarkdownPdf(toc_level=2)
    
    with open("thesis.md", "r", encoding="utf-8") as f:
        md_content = f.read()

    # Add the markdown content as a section
    pdf.add_section(Section(md_content))
    
    # Save the PDF
    pdf.meta["title"] = "Multi-Layer Depth Peeling with Residual Safety Nets"
    pdf.meta["author"] = "MTech Thesis"
    
    out_name = "MTech_Thesis_CubeMeshCompression.pdf"
    pdf.save(out_name)
    print(f"Successfully generated {out_name}")

if __name__ == "__main__":
    generate_pdf()
