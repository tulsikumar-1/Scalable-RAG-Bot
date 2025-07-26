from io import BytesIO
from loader import load_pdf  # Adjust the import path if needed

def test_load_pdf():
    # Load a real PDF file from disk
    with open("/home/tulsi/Downloads/my-rag/data/example.pdf", "rb") as real_pdf:
        
        pdf_bytes = real_pdf.read()

    # Wrap bytes in a BytesIO stream to simulate an uploaded file
    fake_uploaded_file = BytesIO(pdf_bytes)

    # Pass to your function
    extracted_text = load_pdf(fake_uploaded_file)

    # Print result
    print(extracted_text)

    # Optionally test for known content
    assert "expected snippet" in extracted_text

# Run the test
test_load_pdf()

