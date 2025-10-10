
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
        page.get_by_role("link", name="Diagnostic Assistant").click()
        page.wait_for_load_state('networkidle')

        # Use a more robust locator for the file uploader
        bin_file_uploader = page.locator(".stFileUploader:has-text('Upload .bin tune file')")
        bin_file_path = "jules-scratch/verification/dummy.bin"
        bin_file_uploader.locator("input[type='file']").set_input_files(bin_file_path)

        # Select the 'Other' firmware option
        page.get_by_label("Other").check()

        # Use a more robust locator for the XDF file uploader
        xdf_file_uploader = page.locator(".stFileUploader:has-text('Upload .xdf definition file')")
        xdf_file_path = "jules-scratch/verification/dummy.xdf"
        xdf_file_uploader.locator("input[type='file']").set_input_files(xdf_file_path)

        # Enter a question
        page.get_by_label("Enter your diagnostic question:").fill("What is MAF_COR?")

        # Click the button
        page.get_by_role("button", name="Get Diagnostic Answer").click()

        # Wait for the "Thinking Process" expander to appear and take a screenshot
        thinking_process_expander = page.locator("h3:has-text(\"Assistant's Thinking Process\")")
        expect(thinking_process_expander).to_be_visible(timeout=30000)

        page.screenshot(path="jules-scratch/verification/verification.png")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("----------------- PAGE HTML -----------------")
        print(page.content())
        print("----------------- END PAGE HTML -----------------")
        page.screenshot(path="jules-scratch/verification/error.png")

    finally:
        context.close()
        browser.close()

with sync_playwright() as playwright:
    run_verification(playwright)
