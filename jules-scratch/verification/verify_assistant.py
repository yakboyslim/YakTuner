
import re
from playwright.sync_api import sync_playwright, expect

def run_verification(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    try:
        page.goto("http://localhost:8501/", timeout=90000)
        page.wait_for_load_state('networkidle')

        # Navigate to the correct page
        page.locator("a[href='Diagnostic_Assistant']").click()
        page.wait_for_load_state('networkidle')

        # Upload the bin file
        bin_file_uploader = page.locator(".stFileUploader:has-text('Upload .bin tune file')")
        bin_file_path = "jules-scratch/verification/dummy.bin"
        bin_file_uploader.locator("input[type='file']").set_input_files(bin_file_path)

        # Enter the trigger question
        page.get_by_label("Enter your diagnostic question:").fill("What is MAF_COR?")

        # Click the button and take a screenshot of the status updates
        page.get_by_role("button", name="Get Diagnostic Answer").click()

        # Wait for the status to show the RAG step
        expect(page.locator("text=Retrieving context from knowledge base (RAG)...")).to_be_visible(timeout=10000)
        page.screenshot(path="jules-scratch/verification/verification.png")

    except Exception as e:
        print(f"An error occurred: {e}")
        page.screenshot(path="jules-scratch/verification/error.png")

    finally:
        context.close()
        browser.close()

with sync_playwright() as playwright:
    run_verification(playwright)
