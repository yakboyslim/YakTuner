from playwright.sync_api import sync_playwright, expect

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the Streamlit app
        page.goto("http://localhost:8501")

        # Wait for the main app to load. We can look for a known element.
        expect(page.get_by_text("☁️ YAKtuner Online")).to_be_visible(timeout=20000)

        # Scroll to the bottom of the page to ensure the new section is visible
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

        # Find the "Interactive Diagnostic Assistant" header
        assistant_header = page.get_by_role("heading", name="💡 Interactive Diagnostic Assistant")
        expect(assistant_header).to_be_visible()

        # Take a screenshot of the new section
        # We can screenshot the entire page as it's a simple layout
        page.screenshot(path="jules-scratch/verification/diagnostic_assistant_ui.png")

        browser.close()

if __name__ == "__main__":
    run_verification()